"""Self-identifying steering model for Honda/Acura.

Honda lateral control in openpilot is a per-platform, hand tuned affair: every car in
``values.py`` carries its own ``lateralParams.torqueBP/torqueV``, its own ``kpV/kiV``,
a shared guess at ``kf`` and a single global ``steerActuatorDelay``. That scales poorly
across the Honda/Acura platforms in ``opendbc/car/tests/routes.py``, which span
four different EPS/camera generations (Nidec, Bosch, Bosch radarless, Bosch CAN FD),
CAN torque scales from 1000 to 32767 counts, curb masses from 890 kg (N-Box) to
2200 kg (Pilot/MDX), and steer ratios from 10.6 to 21.

This module replaces the guessing with measurement. It observes the car while
openpilot steers it and identifies, online and per fingerprint:

  * ``lat_accel_factor``  - m/s^2 of lateral acceleration per unit of commanded torque,
    as a function of speed (Honda EPS assist is strongly speed dependent, especially on
    Nidec).
  * ``friction``          - the torque needed just to break the steering rack loose.
  * ``offset``            - steady state bias (road crown, alignment, EPS trim).
  * ``asymmetry``         - left/right gain difference, common on high-mileage racks.
  * ``actuator_delay``/``response_tau`` - the rack's dead time and first order response
    time constant. Their sum, ``effective_lag``, is the better determined of the two.
  * ``deadzone``/``max_useful_torque`` - where the EPS actually starts and stops
    responding, i.e. the real usable fraction of ``STEER_MAX``.
  * ``steer_ratio``/``understeer_gradient`` - when an independent yaw source is present.
  * ``driver_torque_threshold`` - the per-car override threshold currently hardcoded in
    ``values.py::STEER_THRESHOLD``.

Everything starts from a per-platform prior derived from the car's own ``CarParams``
(see :func:`prior_from_car_params`), so an unlearned car behaves exactly like today's
tune, and the learned model is blended in as confidence grows.

The module is pure Python + numpy and has no openpilot dependencies, so it can be
driven live from ``CarController`` or offline from logs.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

MODEL_VERSION = 1

# --- sample gating -------------------------------------------------------------------
MIN_LEARN_SPEED = 8.0          # m/s, below this Honda EPS assist is too nonlinear to fit
MAX_LAT_ACCEL = 4.5            # m/s^2, above this the tires are the limit, not the EPS
MAX_LAT_JERK = 8.0             # m/s^3, reject transients we cannot model
# The static terms (gain, friction, offset) and the lag term are fitted on different
# samples. Coulomb friction and a first order lag look nearly identical over smooth
# steering - sign(a_dot) and a_dot are strongly correlated - so fitting them together is
# ill conditioned and splits the truth arbitrarily between them. Fitting the static terms
# only where the car is nearly settled, and the lag term only where it is clearly moving,
# keeps both well determined.
STEADY_JERK = 0.5              # m/s^3, at or below this a sample is a steady state point
DYNAMIC_JERK = 0.8             # m/s^3, above this a sample carries lag information
JERK_WINDOW = 10               # samples used to differentiate lateral acceleration
RACK_MOTION_DEADBAND = 0.5     # deg/s over which the friction sign term ramps
RACK_RATE_TAU = 0.2            # s, filter on rack speed before taking the friction sign
MIN_SETTLE_TIME = 0.30         # s of continuously active, unsaturated control before use

# --- speed schedule ------------------------------------------------------------------
# Bucket edges chosen to straddle the speeds where Honda EPS assist changes character:
# city (below ~27 mph), suburban, and highway.
SPEED_BUCKET_EDGES = (MIN_LEARN_SPEED, 12.0, 18.0, 26.0, 1e3)
LAT_ACCEL_BUCKET_EDGES = (-MAX_LAT_ACCEL, -1.0, -0.3, 0.3, 1.0, MAX_LAT_ACCEL)

MIN_POINTS_PER_BUCKET = 60     # per (speed x lat accel) cell before a bucket is trusted
MIN_TOTAL_POINTS = 600         # before the model as a whole reports ``valid``

FORGETTING_FACTOR = 0.99995    # ~20 min half life at 100 Hz: tracks tire/temperature drift

# --- delay estimation ----------------------------------------------------------------
# The dead time is identified by fitting the same model at a bank of candidate delays and
# keeping the one that explains the command best. Cross correlating command with steering
# rate is cheaper but falls apart on the smooth, low excitation steering openpilot does.
MAX_DELAY = 0.45               # s, Honda Bosch CAN FD racks are the slowest we have seen
MIN_DELAY = 0.03
DELAY_CANDIDATES = tuple(np.round(np.arange(MIN_DELAY, MAX_DELAY + 1e-9, 0.03), 3))
DELAY_BANK_DECIMATION = 5      # only every Nth usable sample updates the bank
DELAY_RESIDUAL_TAU = 2000.0    # samples of residual averaging per candidate

# --- response (deadzone / saturation) profiling --------------------------------------
N_RESPONSE_BINS = 10
RESPONSE_ACTIVE_FRAC = 0.20    # fraction of peak incremental gain that counts as "responding"
RESPONSE_REFRESH = 100         # samples between refreshes of the usable torque range


def _bucket_index(value: float, edges) -> int:
  """Index of the bucket ``value`` falls in, or -1 if it is outside ``edges``."""
  if value < edges[0] or value >= edges[-1]:
    return -1
  return int(np.searchsorted(edges, value, side='right') - 1)


def _smooth_sign(x: float, width: float) -> float:
  """Sign with a linear region, so friction feedforward does not chatter around zero."""
  if width <= 0.0:
    return float(np.sign(x))
  return float(np.clip(x / width, -1.0, 1.0))


@dataclass
class HondaSteerSample:
  """One control step's worth of observation.

  ``torque_cmd`` is the *normalized* command openpilot produced, in [-1, 1] (i.e. the
  value before it is scaled by ``STEER_MAX``), so learned parameters are comparable
  across platforms with wildly different CAN torque scales.
  """
  t: float
  v_ego: float
  torque_cmd: float
  steering_angle_deg: float
  steering_rate_deg: float
  driver_torque: float = 0.0
  lat_active: bool = False
  steering_pressed: bool = False
  saturated: bool = False
  # Optional, from the localizer when available. Without them the learner falls back to
  # the kinematic estimate from steering angle and speed, and skips steer ratio learning.
  lat_accel: float | None = None
  yaw_rate: float | None = None
  roll: float = 0.0


@dataclass
class HondaSteeringModel:
  """The identified plant. Serializable, one per fingerprint."""
  fingerprint: str = ""
  version: int = MODEL_VERSION

  # lat accel (m/s^2) produced per unit of normalized torque, versus speed
  lat_accel_factor_bp: list[float] = field(default_factory=list)
  lat_accel_factor_v: list[float] = field(default_factory=list)
  friction: float = 0.0
  offset: float = 0.0
  asymmetry: float = 0.0            # relative gain difference, right minus left

  response_tau: float = 0.15       # s, first order rack/EPS response time constant
  actuator_delay: float = 0.1
  deadzone: float = 0.0
  max_useful_torque: float = 1.0

  steer_ratio: float = 0.0          # 0 when not learned
  understeer_gradient: float = 0.0  # rad of extra steer per m/s^2, at the road wheel
  driver_torque_threshold: float = 1200.0

  # bookkeeping
  points: int = 0
  learned_buckets: int = 0
  valid: bool = False

  @property
  def effective_lag(self) -> float:
    """Total command-to-motion lag.

    Dead time and the first order response time constant trade off against each other in
    identification (a slower rack looks like a later one over a few seconds of smooth
    steering), so their sum is far better determined than either alone. Use this when you
    care about how late the car is, and ``actuator_delay`` only where a pure dead time is
    required.
    """
    return self.actuator_delay + self.response_tau

  def lat_accel_factor(self, v_ego: float) -> float:
    if not self.lat_accel_factor_v:
      return 1.0
    return float(np.interp(v_ego, self.lat_accel_factor_bp, self.lat_accel_factor_v))

  def to_dict(self) -> dict:
    return asdict(self)

  @staticmethod
  def from_dict(d: dict) -> "HondaSteeringModel":
    if d.get("version") != MODEL_VERSION:
      raise ValueError(f"unsupported honda steering model version {d.get('version')}")
    known = {f for f in HondaSteeringModel.__dataclass_fields__}
    return HondaSteeringModel(**{k: v for k, v in d.items() if k in known})

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), sort_keys=True)

  @staticmethod
  def from_json(s: str | bytes) -> "HondaSteeringModel":
    return HondaSteeringModel.from_dict(json.loads(s))


def prior_from_car_params(CP) -> HondaSteeringModel:
  """Seed a model from what the platform already tells us about itself.

  Every term here is derived from data that already exists per platform, so each of the
  Honda/Acura variants starts somewhere sensible and unique:

  * the EPS torque scale (``lateralParams.torqueBP[-1]``) sets how much authority one
    unit of normalized command has,
  * curb mass and wheelbase set how much lateral acceleration that authority buys,
  * the existing hand tune (``kpV``) is used as a weak sanity anchor,
  * the EPS generation flags set the delay prior.
  """
  steer_max = float(CP.lateralParams.torqueBP[-1]) if len(CP.lateralParams.torqueBP) else 4096.0
  mass = float(CP.mass) or 1500.0
  fingerprint = str(CP.carFingerprint)

  # Self aligning torque scales with the lateral force the tires make, i.e. with mass, so
  # a heavier car makes less lateral acceleration for the same rack effort. The CAN count
  # scale (1000 on a CR-V, 32767 on a Taiwan Odyssey) is a reporting resolution, not an
  # authority, so it only nudges the prior: full normalized command is worth roughly the
  # same on every Honda EPS. 1500 kg is the Accord/Civic middle of the range, where full
  # command is worth about 2.5 m/s^2.
  scale_nudge = float(np.clip((max(steer_max, 1.0) / 4096.0) ** 0.15, 0.8, 1.25))
  lat_accel_factor = float(np.clip(2.5 * (1500.0 / mass) * scale_nudge, 0.6, 6.0))

  delay = float(getattr(CP, "steerActuatorDelay", 0.1)) or 0.1

  return HondaSteeringModel(
    fingerprint=fingerprint,
    lat_accel_factor_bp=[MIN_LEARN_SPEED, 35.0],
    lat_accel_factor_v=[lat_accel_factor, lat_accel_factor],
    friction=0.05,
    offset=0.0,
    asymmetry=0.0,
    response_tau=0.15,
    actuator_delay=float(np.clip(delay, MIN_DELAY, MAX_DELAY)),
    deadzone=0.0,
    max_useful_torque=1.0,
    steer_ratio=float(CP.steerRatio),
    understeer_gradient=0.0,
    driver_torque_threshold=1200.0,
    valid=False,
  )


class _RLS:
  """Recursive least squares with a forgetting factor and an explicit prior.

  Fits ``y = x . theta``. The prior covariance encodes how strongly the seed values are
  believed: small ``p0`` keeps the prior, large ``p0`` lets data take over immediately.
  """

  def __init__(self, theta0, p0):
    self.theta = np.array(theta0, dtype=float)
    self.n = len(self.theta)
    self.P = np.eye(self.n) * np.asarray(p0, dtype=float)
    self.points = 0

  def update(self, x, y: float, lam: float = FORGETTING_FACTOR) -> None:
    x = np.asarray(x, dtype=float)
    Px = self.P @ x
    denom = lam + float(x @ Px)
    if denom < 1e-9:
      return
    K = Px / denom
    self.theta = self.theta + K * (y - float(x @ self.theta))
    self.P = (self.P - np.outer(K, Px)) / lam
    # keep the covariance symmetric and bounded so a long quiet highway stretch cannot
    # let it blow up and then over-react to the first curve
    self.P = np.clip((self.P + self.P.T) / 2.0, -1e3, 1e3)
    self.points += 1


class _DelayBank:
  """Identifies the actuator dead time by model fit rather than by correlation.

  One copy of the steady state model is fitted per candidate delay; the candidate whose
  fit explains the commanded torque best is the car's dead time. This also hands the
  winning fit back as the car's torque model, so the gain, friction and lag terms are
  never contaminated by a misaligned delay.
  """

  def __init__(self, theta0, p0, prior_delay: float, dt: float):
    self.dt = dt
    self.candidates = [d for d in DELAY_CANDIDATES]
    self.rls = [_RLS(theta0, p0) for _ in self.candidates]
    self.residual = np.full(len(self.candidates), np.nan)
    self.n = 0
    self.best = int(np.argmin(np.abs(np.array(self.candidates) - prior_delay)))
    self.delay = float(np.clip(prior_delay, MIN_DELAY, MAX_DELAY))

  def update(self, x, cmd_at) -> None:
    """``cmd_at(delay)`` returns the command issued ``delay`` seconds ago, or None."""
    self.n += 1
    if self.n % DELAY_BANK_DECIMATION:
      return
    k = 1.0 / min(self.n / DELAY_BANK_DECIMATION, DELAY_RESIDUAL_TAU)
    for i, d in enumerate(self.candidates):
      y = cmd_at(d)
      if y is None:
        continue
      err = (y - float(x @ self.rls[i].theta)) ** 2
      self.residual[i] = err if np.isnan(self.residual[i]) else self.residual[i] + k * (err - self.residual[i])
      self.rls[i].update(x, y)

    if np.isnan(self.residual).all():
      return
    self.best = int(np.nanargmin(self.residual))
    self.delay = self._interpolated()

  def _interpolated(self) -> float:
    """Parabolic fit through the winning residual and its neighbours, for sub-grid resolution."""
    i = self.best
    d = self.candidates[i]
    if 0 < i < len(self.candidates) - 1:
      a, b, c = self.residual[i - 1], self.residual[i], self.residual[i + 1]
      if np.isfinite([a, b, c]).all():
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
          step = self.candidates[i + 1] - self.candidates[i]
          d += float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0)) * step
    return float(np.clip(d, MIN_DELAY, MAX_DELAY))

  @property
  def winner(self) -> _RLS:
    return self.rls[self.best]


class _ResponseProfiler:
  """Bins incremental EPS response by |command| to find the deadzone and saturation."""

  def __init__(self):
    self.num = np.zeros(N_RESPONSE_BINS)
    self.den = np.zeros(N_RESPONSE_BINS)
    self.edges = np.linspace(0.0, 1.0, N_RESPONSE_BINS + 1)

  def update(self, torque_cmd: float, d_cmd: float, d_rate: float) -> None:
    if abs(d_cmd) < 1e-4:
      return
    i = min(int(abs(torque_cmd) * N_RESPONSE_BINS), N_RESPONSE_BINS - 1)
    # sign-matched incremental gain: how much rate change one unit of command change
    # buys. Motion against the command counts negative, so a dead or saturated bin
    # cannot look responsive just because the wheel was moving.
    signed = abs(d_rate) if np.sign(d_cmd) == np.sign(d_rate) else -abs(d_rate)
    self.num[i] += signed
    self.den[i] += abs(d_cmd)

  def solve(self) -> tuple[float, float] | None:
    seen = self.den > 0.02
    if seen.sum() < 3:
      return None
    gain = np.zeros(N_RESPONSE_BINS)
    gain[seen] = self.num[seen] / self.den[seen]
    peak = gain.max()
    if peak <= 0.0:
      return None
    responding = np.where(seen & (gain > RESPONSE_ACTIVE_FRAC * peak))[0]
    if not len(responding):
      return None
    deadzone = float(self.edges[responding[0]])
    max_useful = float(self.edges[responding[-1] + 1])
    return deadzone, max_useful


class _OverrideThresholdEstimator:
  """Learns the driver torque noise floor, i.e. ``values.py::STEER_THRESHOLD`` per car."""

  MIN_THRESHOLD = 300.0
  MAX_THRESHOLD = 2000.0

  def __init__(self, prior: float):
    self.threshold = prior
    self.abs_mean = 0.0
    self.abs_var = 0.0
    self.points = 0

  def update(self, driver_torque: float) -> None:
    a = abs(float(driver_torque))
    self.points += 1
    k = 1.0 / min(self.points, 20000)
    self.abs_mean += k * (a - self.abs_mean)
    self.abs_var += k * ((a - self.abs_mean) ** 2 - self.abs_var)
    if self.points > 2000:
      # hands-off torque is roughly zero mean sensor noise; a 6-sigma gate separates it
      # from a real nudge without tripping on rack chatter
      t = self.abs_mean + 6.0 * math.sqrt(max(self.abs_var, 0.0))
      self.threshold = float(np.clip(t, self.MIN_THRESHOLD, self.MAX_THRESHOLD))


class HondaSteeringLearner:
  """Online identification of one Honda's steering.

  Feed it a :class:`HondaSteerSample` every control step; read :meth:`model` whenever you
  want the current best estimate. All estimates start at the platform prior and move only
  on data that passes the gating in :meth:`update`.
  """

  def __init__(self, CP, dt: float = 0.01, prior: HondaSteeringModel | None = None,
               learned: HondaSteeringModel | None = None):
    self.dt = dt
    self.prior = prior if prior is not None else prior_from_car_params(CP)
    self.wheelbase = float(CP.wheelbase)
    self.fingerprint = self.prior.fingerprint

    seed = learned if learned is not None else self.prior
    self.steer_ratio = seed.steer_ratio or float(CP.steerRatio)
    self.understeer_gradient = seed.understeer_gradient

    # Full model, theta = [1/K, friction, offset, asymmetry, tau/K] for y = torque_cmd.
    # Only the delay bank fits all five at once - it just needs to compare candidates, and
    # a split it gets wrong between friction and lag does not change which delay wins.
    inv_k = 1.0 / max(seed.lat_accel_factor(20.0), 1e-3)
    theta0 = [inv_k, seed.friction, seed.offset, seed.asymmetry, seed.response_tau * inv_k]
    # a confident prior on gain and friction, a loose one on the bias terms
    p0 = [0.5 * inv_k ** 2, 0.02, 0.05, 0.05, 0.05 * inv_k ** 2]
    self.delay_bank = _DelayBank(theta0, p0, seed.actuator_delay, dt)
    # static model: theta = [1/K, friction, offset, asymmetry], fitted on settled samples
    self.steady_rls = _RLS(theta0[:4], p0[:4])
    self.speed_rls = [_RLS(theta0[:4], p0[:4]) for _ in range(len(SPEED_BUCKET_EDGES) - 1)]
    # lag model: theta = [tau/K], fitted on moving samples with the static terms removed
    self.lag_rls = _RLS([theta0[4]], [p0[4]])
    self.bucket_counts = np.zeros((len(SPEED_BUCKET_EDGES) - 1,
                                   len(LAT_ACCEL_BUCKET_EDGES) - 1), dtype=int)

    # steer ratio: theta = [steer_ratio, steer_ratio * understeer_gradient]
    self.sr_rls = _RLS([self.steer_ratio, self.steer_ratio * 0.002],
                       [4.0, 0.01])

    self.response = _ResponseProfiler()
    self.override_est = _OverrideThresholdEstimator(seed.driver_torque_threshold)

    self._active_for = 0.0
    self._last = None
    # lateral acceleration is differentiated over a window, not sample to sample: at
    # 100 Hz a sample-to-sample difference is almost entirely sensor noise, and that
    # noise would attenuate the lag term it feeds
    self._accel_hist = deque(maxlen=JERK_WINDOW + 1)
    self._rate_filt = 0.0
    # commands are aligned to the motion they caused using the learned dead time, so the
    # transport delay does not leak into the friction and gain estimates
    self._cmd_history = deque(maxlen=int(MAX_DELAY / dt) + 2)
    self._max_useful_cache = self.prior.max_useful_torque
    self._response_age = 0
    self.points = 0

  # -- helpers ------------------------------------------------------------------------
  def _cmd_at(self, delay: float):
    """The command issued ``delay`` seconds ago, or None if we have not run that long."""
    n = int(round(delay / self.dt))
    if len(self._cmd_history) <= n:
      return None
    return self._cmd_history[len(self._cmd_history) - 1 - n]

  def _lat_accel(self, s: HondaSteerSample) -> float:
    """Measured lateral acceleration, roll compensated, from the best source available."""
    if s.lat_accel is not None:
      a = s.lat_accel
    elif s.yaw_rate is not None:
      a = s.yaw_rate * s.v_ego
    else:
      curvature = math.radians(s.steering_angle_deg) / (self.steer_ratio * self.wheelbase)
      a = curvature * s.v_ego ** 2
    return float(a - math.sin(s.roll) * 9.81)

  # -- main ---------------------------------------------------------------------------
  def update(self, s: HondaSteerSample) -> None:
    last, self._last = self._last, s

    if not s.lat_active or s.steering_pressed:
      self._active_for = 0.0
      self._accel_hist.clear()
      self._rate_filt = 0.0
      self._cmd_history.clear()
      # a disengaged sample is still the best place to measure the driver torque floor,
      # but only when the driver is genuinely not touching the wheel
      if not s.steering_pressed:
        self.override_est.update(s.driver_torque)
      return

    self.override_est.update(s.driver_torque)

    if s.saturated:
      self._active_for = 0.0
    else:
      self._active_for += self.dt

    self._rate_filt += (s.steering_rate_deg - self._rate_filt) * self.dt / RACK_RATE_TAU
    self._cmd_history.append(s.torque_cmd)
    if last is not None:
      self.response.update(s.torque_cmd, s.torque_cmd - last.torque_cmd,
                           s.steering_rate_deg - last.steering_rate_deg)

    lat_accel = self._lat_accel(s)
    self._accel_hist.append(lat_accel)
    lat_jerk = 0.0
    if len(self._accel_hist) == self._accel_hist.maxlen:
      lat_jerk = (self._accel_hist[-1] - self._accel_hist[0]) / (JERK_WINDOW * self.dt)

    # steer ratio needs an independent yaw measurement; the kinematic fallback is circular
    if (s.yaw_rate is not None or s.lat_accel is not None) and s.v_ego > MIN_LEARN_SPEED:
      yaw = s.yaw_rate if s.yaw_rate is not None else lat_accel / max(s.v_ego, 1.0)
      x = np.array([yaw * self.wheelbase / max(s.v_ego, 1.0), lat_accel])
      self.sr_rls.update(x, math.radians(s.steering_angle_deg))
      sr = float(np.clip(self.sr_rls.theta[0], 8.0, 25.0))
      self.steer_ratio = sr
      self.understeer_gradient = float(np.clip(self.sr_rls.theta[1] / max(sr, 1e-3), -0.05, 0.05))

    # --- gating for the torque model ---
    if self._active_for < MIN_SETTLE_TIME:
      return
    if s.v_ego < MIN_LEARN_SPEED or abs(lat_accel) > MAX_LAT_ACCEL:
      return
    if abs(lat_jerk) > MAX_LAT_JERK:
      return
    torque_cmd = self._cmd_at(self.delay_bank.delay)
    # a command the rack could not actually follow tells us nothing about the plant: the
    # motion it produced belongs to a smaller command than the one we recorded
    if torque_cmd is None or abs(torque_cmd) >= self._max_useful() - 1e-3:
      return

    i_speed = _bucket_index(s.v_ego, SPEED_BUCKET_EDGES)
    i_accel = _bucket_index(lat_accel, LAT_ACCEL_BUCKET_EDGES)
    if i_speed < 0 or i_accel < 0:
      return

    # Friction opposes rack motion, so its sign follows a *filtered* rack speed: the raw
    # 100 Hz steering rate is noisy enough that its sign is close to a coin flip when the
    # wheel is barely moving, and a regressor that noisy attenuates the friction estimate
    # toward zero. The linear region is narrow on top of that: too wide and gentle
    # steering never reaches full sign, too narrow and the remaining noise dithers it.
    sign = _smooth_sign(self._rate_filt, RACK_MOTION_DEADBAND)
    x = np.array([lat_accel, sign, 1.0, max(lat_accel, 0.0), lat_jerk])
    self.delay_bank.update(x, self._cmd_at)

    if abs(lat_jerk) <= STEADY_JERK:
      x_static = x[:4]
      self.steady_rls.update(x_static, torque_cmd)
      self.speed_rls[i_speed].update(x_static, torque_cmd)
      self.bucket_counts[i_speed, i_accel] += 1
      self.points += 1
    elif abs(lat_jerk) >= DYNAMIC_JERK:
      # everything the static model already explains is subtracted, so what is left for
      # the lag term to explain is the part of the command that leads the motion
      residual = torque_cmd - float(x[:4] @ self.steady_rls.theta)
      self.lag_rls.update(np.array([lat_jerk]), residual)

  def _max_useful(self) -> float:
    """Largest command worth learning from, refreshed occasionally rather than per sample."""
    if self._response_age <= 0:
      profile = self.response.solve()
      self._max_useful_cache = profile[1] if profile is not None else self.prior.max_useful_torque
      self._response_age = RESPONSE_REFRESH
    self._response_age -= 1
    return self._max_useful_cache

  # -- output -------------------------------------------------------------------------
  def _factor_from(self, rls: _RLS) -> float:
    inv_k = rls.theta[0]
    if inv_k < 1e-3:
      return self.prior.lat_accel_factor(20.0)
    return float(np.clip(1.0 / inv_k, 0.3, 8.0))

  def model(self) -> HondaSteeringModel:
    m = HondaSteeringModel(
      fingerprint=self.fingerprint,
      friction=float(np.clip(self.steady_rls.theta[1], 0.0, 0.4)),
      offset=float(np.clip(self.steady_rls.theta[2], -0.3, 0.3)),
      asymmetry=float(np.clip(self.steady_rls.theta[3], -0.5, 0.5)),
      response_tau=float(np.clip(self.lag_rls.theta[0] * self._factor_from(self.steady_rls),
                                 0.0, 0.6)),
      actuator_delay=self.delay_bank.delay,
      steer_ratio=self.steer_ratio,
      understeer_gradient=self.understeer_gradient,
      driver_torque_threshold=self.override_est.threshold,
      points=self.points,
    )

    # speed schedule: use a bucket's own fit once it has seen enough of the lat accel
    # range, otherwise fall back to the global fit so the schedule stays monotonic-ish
    bps, vs, learned_buckets = [], [], 0
    for i in range(len(SPEED_BUCKET_EDGES) - 1):
      lo, hi = SPEED_BUCKET_EDGES[i], SPEED_BUCKET_EDGES[i + 1]
      centre = lo if i == 0 else (lo + min(hi, 40.0)) / 2.0
      cells = self.bucket_counts[i]
      if (cells >= MIN_POINTS_PER_BUCKET).sum() >= 2:
        vs.append(self._factor_from(self.speed_rls[i]))
        learned_buckets += 1
      else:
        vs.append(self._factor_from(self.steady_rls))
      bps.append(centre)
    m.lat_accel_factor_bp = bps
    m.lat_accel_factor_v = vs
    m.learned_buckets = learned_buckets

    profile = self.response.solve()
    if profile is not None:
      m.deadzone, m.max_useful_torque = profile
    else:
      m.deadzone, m.max_useful_torque = self.prior.deadzone, self.prior.max_useful_torque

    m.valid = self.points >= MIN_TOTAL_POINTS and learned_buckets >= 1
    return m
