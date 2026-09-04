"""A small Honda steering plant, used to exercise the learner across every platform.

The plant is the inverse of the model the learner fits: a first order lag from commanded
torque to lateral acceleration, with a transport delay, Coulomb friction, a steady state
bias and a saturation. Truth values are derived per platform from
``CarSpecs`` and the platform's own EPS torque scale, so a sweep over ``values.py::CAR``
covers the same spread of cars that ``opendbc/car/tests/routes.py`` covers.
"""

from __future__ import annotations

import math
import zlib
from collections import deque
from dataclasses import dataclass

import numpy as np

FRICTION_DIRECTION_TAU = 0.2  # s, rack speed filter behind the Coulomb friction sign


@dataclass
class HondaPlantTruth:
  lat_accel_factor: float
  friction: float
  offset: float
  delay: float
  tau: float
  max_torque: float
  steer_ratio: float
  wheelbase: float

  @staticmethod
  def for_car(CP, seed: int = 0) -> HondaPlantTruth:
    """Plausible-but-not-prior truth for one platform, deterministic in ``seed``."""
    rng = np.random.default_rng(zlib.crc32(f"{CP.carFingerprint}:{seed}".encode()))
    # Same shape as the prior, so the truth stays a plausible Honda rack; what makes this
    # a test of learning is the displacement from the prior below, not a different formula.
    # (Scaling authority with the square root of the CAN count scale, as an earlier version
    # did, made small-count platforms like the CR-V EU absurdly weak: 0.7 m/s^2 at full
    # command, which no Honda rack is.)
    steer_max = float(CP.lateralParams.torqueBP[-1]) or 4096.0
    nudge = float(np.clip((max(steer_max, 1.0) / 4096.0) ** 0.15, 0.8, 1.25))
    base = 2.5 * (1500.0 / float(CP.mass)) * nudge
    return HondaPlantTruth(
      # the point of learning: truth is up to 40% away from what the prior guesses
      lat_accel_factor=float(np.clip(base * rng.uniform(0.6, 1.4), 0.5, 7.0)),
      friction=float(rng.uniform(0.01, 0.09)),
      offset=float(rng.uniform(-0.05, 0.05)),
      delay=float(rng.uniform(0.04, 0.20)),
      tau=float(rng.uniform(0.08, 0.22)),
      max_torque=float(rng.uniform(0.85, 1.0)),
      steer_ratio=float(CP.steerRatio),
      wheelbase=float(CP.wheelbase),
    )


class HondaPlant:
  def __init__(self, truth: HondaPlantTruth, dt: float = 0.01):
    self.truth = truth
    self.dt = dt
    self.lat_accel = 0.0
    self.angle_deg = 0.0
    self.rate_deg = 0.0
    self.direction = 0.0
    self.rate_filt = 0.0
    self.buf = deque([0.0] * max(1, int(round(truth.delay / dt))),
                     maxlen=max(1, int(round(truth.delay / dt))))

  def step(self, torque_cmd: float, v_ego: float) -> None:
    t = self.truth
    self.buf.append(float(np.clip(torque_cmd, -t.max_torque, t.max_torque)))
    u = self.buf[0]
    # Coulomb friction opposes rack motion, so it is signed by the steering rate:
    # u = a / K + friction * sign(rack rate) + offset
    # Direction comes from a filtered rack speed. Keying Coulomb friction off the raw
    # per-sample rate makes the discrete plant limit-cycle at Nyquist (the friction term
    # flips sign every step and drives the oscillation it is reacting to), which is a
    # numerical artifact, not something a real rack does.
    self.rate_filt += (self.rate_deg - self.rate_filt) * self.dt / FRICTION_DIRECTION_TAU
    if abs(self.rate_filt) > 0.5:
      self.direction = float(np.sign(self.rate_filt))
    a_ss = (u - t.offset - t.friction * self.direction) * t.lat_accel_factor
    self.lat_accel += (a_ss - self.lat_accel) * self.dt / t.tau

    angle = math.degrees(self.lat_accel / max(v_ego, 1.0) ** 2 * t.steer_ratio * t.wheelbase)
    self.rate_deg = (angle - self.angle_deg) / self.dt
    self.angle_deg = angle
