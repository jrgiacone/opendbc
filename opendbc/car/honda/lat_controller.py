"""Adaptive Honda lateral controller.

Today's Honda tune is a per-platform PID on lateral acceleration error with a fixed
feedforward gain (``lateralTuning.pid.kf = 0.00006`` for every car) and hand picked
``kpV``/``kiV``. This controller keeps the same shape but takes its plant knowledge from
:mod:`opendbc.car.honda.steering_learner` instead of a table:

  * feedforward is ``desired / lat_accel_factor(v)`` using the car's *measured* gain,
    plus its measured friction and steady state offset,
  * feedback gains are specified in plant-independent units and divided by the same
    measured gain, so one set of numbers behaves the same on an N-Box and a Pilot,
  * feedback is PI only, as Honda's PID tune is today: the derivative of an error built
    from a noisy lateral acceleration measurement is almost entirely noise,
  * the command is compensated for the car's measured left/right asymmetry, and clipped
    to the torque range the rack is measured to actually follow,
  * the integrator freezes on saturation, driver override and learned-model resets.

Learned values are blended in over ``BLEND_TIME`` seconds of valid model, so a car that
has not learned yet drives exactly like its prior (i.e. like today).
"""

from __future__ import annotations

import numpy as np

from opendbc.car.honda.steering_learner import (
  HondaSteeringModel,
  HondaSteerSample,
  HondaSteeringLearner,
  prior_from_car_params,
  _smooth_sign,
)

# Feedback gains in plant-independent units: error in m/s^2 in, "m/s^2 of correction" out.
# Dividing by the learned lat_accel_factor turns them into normalized torque.
KP_BP = [0.0, 10.0, 30.0]
KP_V = [0.20, 0.32, 0.42]
KI_BP = [0.0, 10.0, 30.0]
KI_V = [0.08, 0.11, 0.14]

I_LIMIT = 0.35            # normalized torque the integrator alone may command
FRICTION_WIDTH = 0.20     # m/s^2 of desired lat jerk over which friction FF ramps in
BLEND_TIME = 60.0         # s of valid learned model before it is fully trusted


class HondaLatDebug:
  __slots__ = ("feedforward", "p", "i", "output", "saturated", "blend", "lat_accel_factor")

  def __init__(self, **kw):
    for k in self.__slots__:
      setattr(self, k, kw.get(k, 0.0))

  def __repr__(self):
    return "HondaLatDebug(" + ", ".join(f"{k}={getattr(self, k)}" for k in self.__slots__) + ")"


class HondaAdaptiveLatController:
  """PID + learned feedforward on lateral acceleration.

  Call :meth:`update` every control step with the desired and measured lateral
  acceleration; it returns normalized torque in [-1, 1] for
  ``CarController`` to scale by ``STEER_MAX``. The same call feeds the learner, so
  identification happens as a side effect of driving.
  """

  def __init__(self, CP, dt: float = 0.01, learned: HondaSteeringModel | None = None):
    self.dt = dt
    self.CP = CP
    self.prior = prior_from_car_params(CP)
    self.learner = HondaSteeringLearner(CP, dt=dt, prior=self.prior, learned=learned)
    self.model = learned if (learned is not None and learned.valid) else self.prior
    self.blend = 1.0 if (learned is not None and learned.valid) else 0.0

    self.i = 0.0
    self.last_output = 0.0
    self._model_age = BLEND_TIME if self.blend else 0.0
    self._t = 0.0

  # -- model plumbing ------------------------------------------------------------------
  def _refresh_model(self) -> None:
    learned = self.learner.model()
    if learned.valid:
      self.model = learned
      self._model_age += self.dt
    else:
      self._model_age = max(0.0, self._model_age - self.dt)
    self.blend = float(np.clip(self._model_age / BLEND_TIME, 0.0, 1.0))

  def _blended(self, name: str, v_ego: float | None = None) -> float:
    if name == "lat_accel_factor":
      a, b = self.prior.lat_accel_factor(v_ego or 0.0), self.model.lat_accel_factor(v_ego or 0.0)
    else:
      a, b = getattr(self.prior, name), getattr(self.model, name)
    return float(a + self.blend * (b - a))

  def reset(self) -> None:
    self.i = 0.0
    self.last_output = 0.0

  # -- control -------------------------------------------------------------------------
  def update(self, desired_lat_accel: float, measured_lat_accel: float, v_ego: float,
             steering_angle_deg: float, steering_rate_deg: float, lat_active: bool,
             steering_pressed: bool = False, driver_torque: float = 0.0,
             yaw_rate: float | None = None, roll: float = 0.0,
             desired_lat_jerk: float = 0.0) -> tuple[float, HondaLatDebug]:
    self._t += self.dt
    self._refresh_model()

    factor = max(self._blended("lat_accel_factor", v_ego), 1e-3)
    friction = self._blended("friction")
    offset = self._blended("offset")
    asymmetry = self._blended("asymmetry")
    max_torque = self._blended("max_useful_torque")

    if not lat_active:
      self.reset()
      output = 0.0
      debug = HondaLatDebug(blend=self.blend, lat_accel_factor=factor)
    else:
      # feedforward from the measured plant, including the car's own friction and bias.
      # The asymmetry term mirrors the learned model exactly: a rack that is weaker in one
      # direction needs proportionally more command that way.
      ff = desired_lat_accel / factor + offset + asymmetry * max(desired_lat_accel, 0.0)
      # The rack is late by a dead time plus a first order lag, and identification splits
      # that total between the two only weakly. Leading by the total is both what the
      # plant needs and what is actually well determined; it is the same compensation
      # steerActuatorDelay performs today, expressed as a jerk feedforward.
      lead = self._blended("actuator_delay") + self._blended("response_tau")
      ff += lead * desired_lat_jerk / factor
      ff += friction * _smooth_sign(desired_lat_jerk if desired_lat_jerk else desired_lat_accel,
                                    FRICTION_WIDTH)

      error = desired_lat_accel - measured_lat_accel
      kp = float(np.interp(v_ego, KP_BP, KP_V)) / factor
      ki = float(np.interp(v_ego, KI_BP, KI_V)) / factor

      p = kp * error

      unsaturated = ff + p + self.i
      # freeze the integrator when we are already asking for everything the rack has, or
      # when the driver is fighting us: both make the error signal meaningless
      if abs(unsaturated) < max_torque and not steering_pressed:
        self.i = float(np.clip(self.i + ki * error * self.dt, -I_LIMIT, I_LIMIT))

      output = ff + p + self.i

      clipped = float(np.clip(output, -max_torque, max_torque))
      debug = HondaLatDebug(feedforward=ff, p=p, i=self.i, output=clipped,
                            saturated=abs(output) > max_torque, blend=self.blend,
                            lat_accel_factor=factor)
      output = clipped

    self.last_output = output
    self.learner.update(HondaSteerSample(
      t=self._t, v_ego=v_ego, torque_cmd=output,
      steering_angle_deg=steering_angle_deg, steering_rate_deg=steering_rate_deg,
      driver_torque=driver_torque, lat_active=lat_active,
      steering_pressed=steering_pressed, saturated=bool(debug.saturated),
      lat_accel=measured_lat_accel, yaw_rate=yaw_rate, roll=roll,
    ))
    return output, debug

  @property
  def lookahead(self) -> float:
    """Seconds the desired signal should be advanced to cover the rack's dead time."""
    return self._blended("actuator_delay")
