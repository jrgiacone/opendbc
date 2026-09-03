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
  * ``max_useful_torque`` - the usable fraction of ``STEER_MAX``. This one is carried
    from the prior rather than learned: a controller that refuses to overdrive the rack
    never produces the evidence that would show where the rack gives up, and a limit
    guessed from the little evidence it does produce is worse than no limit at all.
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

MODEL_VERSION = 2

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
JERK_WINDOW_S = 0.1            # s over which lateral acceleration is differentiated: long
                               # enough that the difference is signal rather than noise,
                               # short enough not to smear the lag it is used to measure
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

# Resuming from a cached model restores how sure it was, not just what it thought, but not
# quite in full: tires, alignment, load and temperature can all change between ignitions,
# and a learner that resumes at full confidence cannot notice. Inflating the covariance
# re-opens it just enough to move if the car has.
RESTART_COVARIANCE_INFLATION = 2.0

# A gain outside this range is not a measurement, it is a diverged fit. Surfacing that is
# the point: quietly substituting the prior makes a broken fit indistinguishable from an
# unfitted one, which is exactly what hid the divergence on route 729a2e65b1f6201d.
K_MIN_VALID = 0.2
K_MAX_VALID = 8.0

# Asymmetry needs command in both directions to mean anything: with one-signed command,
# max(u, 0) is either identical to u or identically zero, and the pair is unidentifiable.
ASYM_MIN_CMD = 0.05
ASYM_MIN_SAMPLES = 200

# A fit over a narrow band of lateral acceleration is a fit to noise, whatever its point
# count: the signal has to span enough range to outweigh the error on it.
MIN_LAT_ACCEL_SPAN = 1.3

# --- roll compensation ---------------------------------------------------------------
# Road roll enters the yaw-derived lateral acceleration as sin(roll)*9.81 and is supposed
# to be removed before fitting. On gentle lane keeping the estimate is not small enough
# relative to the signal for that to be free. Measured over the 76650 gated samples of
# route 729a2e65b1f6201d/00000010:
#
#   yaw-derived lat accel            sigma 0.207 m/s^2
#   roll compensation                sigma 0.221 m/s^2   (1.07x the signal)
#   corr(lat accel, roll comp)              -0.298
#   corr(torque_cmd, lat accel)             +0.191
#   corr(torque_cmd, lat accel - roll comp) +0.114
#
# Subtracting it raised the target's variance and cost 40% of the correlation with the
# command: on that route the compensation carried more error than road roll. So it is
# applied only where the estimate is quiet enough to believe, and the crown it leaves
# behind is picked up by ``offset``, which is what that term is for. Both correlations
# are tracked and published so the fleet answers this with data rather than with one
# route's numbers.
# Superelevation on ordinary roads tops out near 10%, i.e. sin(0.1)*9.81 = 0.98 m/s^2.
# A compensation larger than that is not road crown, it is the estimate.
MAX_ROLL_COMPENSATION = 1.0
# Road crown changes over seconds, so the estimate moving faster than this is the
# localizer settling rather than the road banking. Measured against a low passed roll:
# the raw estimate arrives at 20 Hz and is held between updates, so differentiating it
# at the learner's rate reads zero on most samples and a spike on the rest.
MAX_ROLL_RATE = 0.05           # rad/s
ROLL_RATE_TAU = 0.5            # s of low pass before the rate is taken
ROLL_CORR_TAU = 20000.0        # samples of averaging on the published correlations

# No direction of the covariance may wind up beyond this multiple of its prior width. An
# unexcited direction otherwise grows until the first sample that touches it lands with
# enormous gain.
P_MAX_SCALE = 4.0
# the delay bank re-races from scratch each drive; until it has run enough candidates to
# mean anything, report the delay we resumed with rather than a half finished race
DELAY_MIN_UPDATES = 200
# How much better than the prior's own candidate the winner has to be before it is
# believed. Fitting (dead time, tau) by output error over route 729a2e65b1f6201d - the
# method that recovers both exactly on a synthetic rack of known truth, at this route's
# noise and excitation - leaves a residual surface with 0.6% total leverage on dead time
# and 0.2% on tau, sliding monotonically into the corner of the grid rather than settling
# anywhere. Against 578% and 224% on the synthetic, that is around a thousand times less
# structure than identification needs, and an argmin over it is a coin toss.
#
# This is a reporting rule and nothing more: the winner is still used, because the delay
# the static fit aligns on is not a free choice. A misaligned command contaminates gain,
# friction and offset - holding the delay at the prior on a plant whose real delay is
# elsewhere inverted the learned speed schedule outright in test_learns_speed_schedule,
# on a phase error too small to show in the residual. So the bank keeps racing and keeps
# aligning on what it finds; delay_learned says whether that winner was ever meaningfully
# better than the prior's own candidate, so a log can tell a measurement from a coin toss
# without the model having to pretend it knows less than it does.
DELAY_MIN_IMPROVEMENT = 0.05

# --- delay estimation ----------------------------------------------------------------
# The dead time is identified by fitting the same model at a bank of candidate delays and
# keeping the one that explains the command best. Cross correlating command with steering
# rate is cheaper but falls apart on the smooth, low excitation steering openpilot does.
MAX_DELAY = 0.45               # s, Honda Bosch CAN FD racks are the slowest we have seen
MIN_DELAY = 0.03
DELAY_CANDIDATES = tuple(np.round(np.arange(MIN_DELAY, MAX_DELAY + 1e-9, 0.03), 3))
DELAY_BANK_DECIMATION = 10     # only every Nth usable sample updates the bank: the bank
                               # is by far the most expensive part of the learner, and the
                               # dead time converges over minutes, not seconds
DELAY_RESIDUAL_TAU = 2000.0    # samples of residual averaging per candidate



def speed_bucket_centres() -> list[float]:
  """The speed each bucket's gain estimate is reported at."""
  out = []
  for i in range(len(SPEED_BUCKET_EDGES) - 1):
    lo, hi = SPEED_BUCKET_EDGES[i], SPEED_BUCKET_EDGES[i + 1]
    out.append(lo if i == 0 else (lo + min(hi, 40.0)) / 2.0)
  return out


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
  # False when the yaw/acceleration source is too stale to believe. The sample still
  # advances the command history - dropping it outright would break the delay alignment,
  # which counts samples - but nothing is fitted to it.
  lat_accel_valid: bool = True


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
  max_useful_torque: float = 1.0

  steer_ratio: float = 0.0          # 0 when not learned
  understeer_gradient: float = 0.0  # rad of extra steer per m/s^2, at the road wheel
  driver_torque_threshold: float = 1200.0

  # bookkeeping. ``points`` is a lifetime count across every drive that fed this model,
  # not a count for the current one.
  points: int = 0
  learned_buckets: int = 0
  valid: bool = False
  # a fit that left the physically possible range, and how often that has happened. Both
  # are published: a diverged fit must be visible, not silently replaced by the prior.
  diverged: bool = False
  resets: int = 0
  asymmetry_learned: bool = False

  # Evidence about roll compensation rather than about the car: what fraction of samples
  # the compensation was trusted on, and how well the command tracks the fitted target
  # with it applied against the uncompensated one. Published so the fleet can settle
  # whether subtracting the roll estimate helps this signal or hurts it.
  # Whether ``actuator_delay`` won its race by enough to be called a measurement, and
  # whether the winning candidate sat on an end of the bank's grid. The delay is published
  # either way, because the fit has to align on something; these say how much to believe
  # it. ``effective_lag`` is the figure to use regardless: it is the sum, and the sum is
  # what the data determines.
  delay_learned: bool = False
  delay_railed: bool = False

  roll_comp_fraction: float = 0.0
  lat_accel_torque_corr: float = 0.0
  lat_accel_torque_corr_raw: float = 0.0

  # Enough state to resume rather than restart. ``bucket_counts`` is which speed and
  # lateral acceleration cells have been covered; ``covariance`` is how sure each fit was.
  # Neither is published: they are for the next ignition, not for reading.
  bucket_counts: list[list[int]] = field(default_factory=list)
  covariance: dict = field(default_factory=dict)

  @property
  def effective_lag(self) -> float:
    """Total command-to-motion lag.

    Dead time and the first order response time constant trade off against each other in
    identification (a slower rack looks like a later one over a few seconds of smooth
    steering), so their sum is far better determined than either alone. On real lane
    keeping the split is not merely worse determined but unobservable - 0.2% residual
    leverage on tau across its whole range on route 729a2e65b1f6201d - so this sum is the
    only lag figure to trust. Use ``actuator_delay`` alone where a pure dead time is
    required, and check ``delay_learned`` before reading it as a measurement.
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
  # FIXME: this reads an unpopulated torque scale as a real, very small one. The guard is
  # on the list being empty, but HONDA_CLARITY carries torqueBP=[0], which is non-empty
  # with a last element of zero, so it takes the branch and gets steer_max=0 rather than
  # the 4096.0 default. max(steer_max, 1.0) below then floors scale_nudge at 0.8 and the
  # Clarity ends up with the lowest gain prior in the fleet (1.520) despite being 276 kg
  # lighter than the Pilot, which is what test_priors_differentiate_platforms is failing
  # on. It is the only platform affected today, but any car whose lateralParams are not
  # filled in inherits the same silently wrong prior rather than the default.
  #
  # Left alone deliberately: the fix changes what "missing" means for every platform's
  # prior, and the priors want re-deriving against real routes anyway - route
  # 729a2e65b1f6201d fits 2.49 against a 2.65 prior on a Civic, which is close, but one
  # route is not evidence about the fleet. Revisit both together.
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
    self.p0 = np.asarray(p0, dtype=float)
    self.P = np.eye(self.n) * self.p0
    self.p_max = self.p0 * P_MAX_SCALE
    self.points = 0

  def reset(self, theta0=None) -> None:
    """Back to the prior. Used when a fit has diverged: a diverged fit cannot recover on
    its own, because every later sample is filtered through the broken estimate."""
    if theta0 is not None:
      self.theta = np.array(theta0, dtype=float)
    self.P = np.eye(self.n) * self.p0

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
    self.P = (self.P + self.P.T) / 2.0
    # cap each direction at a multiple of its prior width: a direction the data never
    # excites otherwise winds up until the first sample touching it moves it violently
    d = np.diag(self.P)
    if np.any(d > self.p_max):
      scale = np.minimum(1.0, self.p_max / np.maximum(d, 1e-12))
      r = np.sqrt(scale)
      self.P = self.P * np.outer(r, r)
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
    self.prior_index = int(np.argmin(np.abs(np.array(self.candidates) - prior_delay)))
    self.best = self.prior_index
    self.delay = float(np.clip(prior_delay, MIN_DELAY, MAX_DELAY))
    self.prior_delay = self.delay
    # the winner sat on an end of the grid: an argmin that ran out of candidates is a
    # statement about the grid, not about the rack, and is published as such
    self.railed = False
    self.learned = False

  def update(self, build_x, y: float) -> None:
    """``build_x(delay)`` returns the regressor using the command issued ``delay`` ago."""
    self.n += 1
    if self.n % DELAY_BANK_DECIMATION:
      return
    k = 1.0 / min(self.n / DELAY_BANK_DECIMATION, DELAY_RESIDUAL_TAU)
    for i, d in enumerate(self.candidates):
      x = build_x(d)
      if x is None:
        continue
      err = (y - float(x @ self.rls[i].theta)) ** 2
      self.residual[i] = err if np.isnan(self.residual[i]) else self.residual[i] + k * (err - self.residual[i])
      self.rls[i].update(x, y)

    # Until enough candidates have been scored on enough samples, the race is noise: on a
    # resumed model that would replace a delay learned over hours with one scored over
    # seconds, and on a fresh one it would swing the estimate around early in a drive.
    if np.isnan(self.residual).all() or self.n < DELAY_MIN_UPDATES:
      return
    best = int(np.nanargmin(self.residual))
    # A flat surface makes the argmin a coin toss between candidates that explain the data
    # equally well, and following it publishes noise as a measurement.
    reference = self.residual[self.prior_index]
    improvement = 0.0 if not np.isfinite(reference) or reference <= 0.0 else \
        1.0 - self.residual[best] / reference
    self.best = best
    self.railed = best in (0, len(self.candidates) - 1)
    self.learned = improvement >= DELAY_MIN_IMPROVEMENT
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


class _StreamingCorrelation:
  """Exponentially weighted correlation between two streams.

  Published as evidence, not used in any fit: it is the only way to tell from a log
  whether a preprocessing step helped or hurt the thing it feeds.
  """

  def __init__(self, tau: float):
    self.tau = tau
    self.n = 0
    self.mx = 0.0
    self.my = 0.0
    self.vx = 0.0
    self.vy = 0.0
    self.cxy = 0.0

  def update(self, x: float, y: float) -> None:
    self.n += 1
    k = 1.0 / min(self.n, self.tau)
    dx, dy = x - self.mx, y - self.my
    self.mx += k * dx
    self.my += k * dy
    self.vx += k * (dx * dx - self.vx)
    self.vy += k * (dy * dy - self.vy)
    self.cxy += k * (dx * dy - self.cxy)

  @property
  def value(self) -> float:
    if self.n < 2 or self.vx <= 0.0 or self.vy <= 0.0:
      return 0.0
    return float(np.clip(self.cxy / math.sqrt(self.vx * self.vy), -1.0, 1.0))


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
    # The fit is oriented with the *command* as the regressor and the *measured* lateral
    # acceleration as the dependent variable, matching torqued. The reverse - which this
    # learner used until route 729a2e65b1f6201d showed what it does - puts the noisiest
    # signal in the system on the regressor side, where its noise biases its own
    # coefficient toward zero. That coefficient was 1/K, so the published gain 1/inv_k
    # was driven upward without limit, to the 8.0 rail and beyond into negative inv_k.
    #
    #   a = K*u - K*offset - K*friction*sign - K*asym*max(u, 0) - tau*a_dot
    #
    # theta = [K, -K*offset, -K*friction, -K*asym] over x = [u, 1, sign, max(u, 0)],
    # so the gain is read out directly, with no reciprocal to amplify anything.
    k0 = max(seed.lat_accel_factor(20.0), 1e-3)
    theta0 = [k0, -k0 * seed.offset, -k0 * seed.friction, -k0 * seed.asymmetry,
              -seed.response_tau]
    p0 = [0.5 * k0 ** 2, 0.05 * k0 ** 2, 0.02 * k0 ** 2, 0.05 * k0 ** 2, 0.02]
    # the bank races the static model only; the lag term is fitted separately
    self.delay_bank = _DelayBank(theta0[:4], p0[:4], seed.actuator_delay, dt)
    # static model: theta = [1/K, friction, offset, asymmetry], fitted on settled samples
    self.steady_rls = _RLS(theta0[:4], p0[:4])
    # Each speed bucket resumes from the seed's gain *at that speed*. Seeding them all
    # from one point on the schedule would flatten it every ignition, throwing away the
    # speed dependence the schedule exists to capture.
    self.speed_rls = []
    for centre in speed_bucket_centres():
      k_i = max(seed.lat_accel_factor(centre), 1e-3)
      self.speed_rls.append(_RLS([k_i, -k_i * seed.offset, -k_i * seed.friction,
                                  -k_i * seed.asymmetry],
                                 [0.5 * k_i ** 2, 0.05 * k_i ** 2, 0.02 * k_i ** 2,
                                  0.05 * k_i ** 2]))
    # lag model: theta = [tau/K], fitted on moving samples with the static terms removed
    self.lag_rls = _RLS([theta0[4]], [p0[4]])
    # asymmetry stays switched off until the command has been driven both ways
    self._asym_enabled = False
    self._asym_counts = [0, 0]
    self.resets = 0
    # roll compensation bookkeeping and the evidence for whether it is worth applying
    self._roll_filt: float | None = None
    self._raw_lat_accel = 0.0
    self.roll_comp_applied = 0
    self.roll_comp_skipped = 0
    self.corr_compensated = _StreamingCorrelation(ROLL_CORR_TAU)
    self.corr_raw = _StreamingCorrelation(ROLL_CORR_TAU)
    self.bucket_counts = np.zeros((len(SPEED_BUCKET_EDGES) - 1,
                                   len(LAT_ACCEL_BUCKET_EDGES) - 1), dtype=int)

    # resume the evidence behind those numbers, not just the numbers
    self.points = 0
    if learned is not None:
      self._resume(learned)

    # steer ratio: theta = [steer_ratio, steer_ratio * understeer_gradient]
    self.sr_rls = _RLS([self.steer_ratio, self.steer_ratio * 0.002],
                       [4.0, 0.01])

    self.override_est = _OverrideThresholdEstimator(seed.driver_torque_threshold)

    self._active_for = 0.0
    self._last = None
    # lateral acceleration is differentiated over a window, not sample to sample: at
    # 100 Hz a sample-to-sample difference is almost entirely sensor noise, and that
    # noise would attenuate the lag term it feeds
    self._jerk_window = max(1, int(round(JERK_WINDOW_S / dt)))
    self._accel_hist = deque(maxlen=self._jerk_window + 1)
    self._rate_filt = 0.0
    # commands are aligned to the motion they caused using the learned dead time, so the
    # transport delay does not leak into the friction and gain estimates
    self._cmd_history = deque(maxlen=int(MAX_DELAY / dt) + 2)

  def _resume(self, learned: HondaSteeringModel) -> None:
    """Restore the covariance and coverage a cached model was built from.

    Without this a resumed learner holds the right numbers with none of the confidence
    behind them, so the first minutes of every drive move them as freely as if the car
    had never been measured, and the model reports itself unconverged until it has
    re-earned coverage it already has.

    Anything missing or misshapen is skipped rather than trusted: a cache is data, and a
    learner that resumes from a corrupt one is worse than one that starts over.
    """
    try:
      counts = np.asarray(learned.bucket_counts, dtype=int) if learned.bucket_counts else None
      if counts is not None and counts.shape == self.bucket_counts.shape:
        self.bucket_counts = counts
        self.points = int(learned.points)
    except (TypeError, ValueError):
      pass

    cov = learned.covariance if isinstance(learned.covariance, dict) else {}
    self._restore_covariance(self.steady_rls, cov.get("steady"))
    self._restore_covariance(self.lag_rls, cov.get("lag"))
    speed_cov = cov.get("speed")
    for rls, P in zip(self.speed_rls, speed_cov if isinstance(speed_cov, list) else [], strict=False):
      self._restore_covariance(rls, P)

  @staticmethod
  def _restore_covariance(rls: _RLS, stored) -> None:
    if stored is None:
      return
    try:
      P = np.asarray(stored, dtype=float)
    except (TypeError, ValueError):
      return
    if P.shape != rls.P.shape or not np.isfinite(P).all():
      return
    P = P * RESTART_COVARIANCE_INFLATION
    # a resumed fit must never claim to be less sure than an unlearned one; that would
    # mean the cache was noise, and the prior is the better starting point
    if np.any(np.diag(P) > np.diag(rls.P)):
      return
    rls.P = P

  # -- helpers ------------------------------------------------------------------------
  def _cmd_at(self, delay: float):
    """The command issued ``delay`` seconds ago, or None if we have not run that long."""
    n = int(round(delay / self.dt))
    if len(self._cmd_history) <= n:
      return None
    return self._cmd_history[len(self._cmd_history) - 1 - n]

  def _roll_compensation(self, s: HondaSteerSample) -> float:
    """The roll term to subtract, or zero where the estimate is not worth believing.

    Two ways for the estimate to be wrong in a way that costs more than the roll it
    removes: too large to be road crown, or moving too fast to be road crown. A settling
    or resetting localizer produces both. What is skipped here is a slowly varying bias,
    which is exactly what ``offset`` absorbs.

    This catches a pathological estimate, not a merely imprecise one. On route
    729a2e65b1f6201d the compensation cost 40% of the correlation between command and
    fitted target while staying inside both limits, and no threshold on magnitude or rate
    separates the harmful samples cleanly - the loss is not monotonic in either. Whether
    the compensation belongs in the fit at all is what the published correlations are
    for; tuning these limits to one route's shape would be fitting the route.
    """
    comp = math.sin(s.roll) * 9.81
    if self._roll_filt is None:
      self._roll_filt = s.roll
    prev, self._roll_filt = self._roll_filt, self._roll_filt + (s.roll - self._roll_filt) * self.dt / ROLL_RATE_TAU
    rate = abs(self._roll_filt - prev) / self.dt
    if abs(comp) > MAX_ROLL_COMPENSATION or rate > MAX_ROLL_RATE:
      self.roll_comp_skipped += 1
      return 0.0
    self.roll_comp_applied += 1
    return comp

  def _lat_accel(self, s: HondaSteerSample) -> float:
    """Measured lateral acceleration, roll compensated, from the best source available."""
    if s.lat_accel is not None:
      a = s.lat_accel
    elif s.yaw_rate is not None:
      a = s.yaw_rate * s.v_ego
    else:
      curvature = math.radians(s.steering_angle_deg) / (self.steer_ratio * self.wheelbase)
      a = curvature * s.v_ego ** 2
    self._raw_lat_accel = float(a)
    return float(a - self._roll_compensation(s))

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

    if not s.lat_accel_valid:
      # a stale yaw rate is a wrong lateral acceleration, and it is the regressor-side
      # error that this model is least able to tolerate
      self._accel_hist.clear()
      return

    lat_accel = self._lat_accel(s)
    self._accel_hist.append(lat_accel)
    lat_jerk = 0.0
    if len(self._accel_hist) == self._accel_hist.maxlen:
      lat_jerk = (self._accel_hist[-1] - self._accel_hist[0]) / (self._jerk_window * self.dt)

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
    if torque_cmd is None:
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

    def build_x(delay: float):
      u = self._cmd_at(delay)
      if u is None:
        return None
      # the asymmetry column duplicates the command column while the command is
      # one-signed, so it stays zeroed until both directions have been seen
      asym = max(u, 0.0) if self._asym_enabled else 0.0
      return np.array([u, 1.0, sign, asym])

    # a command the rack could not actually follow tells us nothing about the plant: the
    # motion it produced belongs to a smaller command than the one we recorded
    if abs(torque_cmd) >= self.prior.max_useful_torque - 1e-3:
      return

    x_static = build_x(self.delay_bank.delay)
    if x_static is None:
      return

    self.delay_bank.update(build_x, lat_accel)

    if abs(lat_jerk) <= STEADY_JERK:
      # the same samples the static model is fitted on, scored both ways: whichever
      # target tracks the command better is the one the fit should be using
      self.corr_compensated.update(torque_cmd, lat_accel)
      self.corr_raw.update(torque_cmd, self._raw_lat_accel)
      self.steady_rls.update(x_static, lat_accel)
      self.speed_rls[i_speed].update(x_static, lat_accel)
      self.bucket_counts[i_speed, i_accel] += 1
      self.points += 1
      self._note_asymmetry_excitation(torque_cmd)
      self._check_divergence()
    elif abs(lat_jerk) >= DYNAMIC_JERK:
      # what the static model cannot explain on a moving sample is the rack's lag
      residual = lat_accel - float(x_static @ self.steady_rls.theta)
      self.lag_rls.update(np.array([lat_jerk]), residual)

  def _note_asymmetry_excitation(self, torque_cmd: float) -> None:
    """Enable the asymmetry column once the command has gone both ways enough times."""
    if self._asym_enabled:
      return
    if torque_cmd > ASYM_MIN_CMD:
      self._asym_counts[0] += 1
    elif torque_cmd < -ASYM_MIN_CMD:
      self._asym_counts[1] += 1
    if min(self._asym_counts) >= ASYM_MIN_SAMPLES:
      self._asym_enabled = True
      # the column starts from nothing, so give it its prior width back rather than
      # whatever the covariance happened to be while it sat unused
      for rls in [self.steady_rls] + self.speed_rls:
        rls.theta[3] = 0.0
        rls.P[3, :] = 0.0
        rls.P[:, 3] = 0.0
        rls.P[3, 3] = rls.p0[3]

  def _check_divergence(self) -> None:
    """A fit that has left the physically possible range is reset, not published.

    Once the gain is wrong every later sample is interpreted through it, so a diverged
    fit does not recover on its own. Resetting costs the drive's learning; leaving it
    costs a model that looks plausible and is not.
    """
    for rls in [self.steady_rls] + self.speed_rls:
      if not (K_MIN_VALID <= rls.theta[0] <= K_MAX_VALID) or not np.isfinite(rls.theta).all():
        rls.reset(theta0=[self.prior.lat_accel_factor(20.0), 0.0, 0.0, 0.0])
        self.resets += 1

  # -- output -------------------------------------------------------------------------
  def _factor_from(self, rls: _RLS) -> float | None:
    """The gain this fit has measured, or None if it has not measured one.

    None means "no answer", and the caller substitutes the prior. Returning the prior
    from here instead - as this did until route 729a2e65b1f6201d - makes a diverged fit
    and an unfitted one look identical in the published model, which is precisely what
    stopped the divergence being visible.
    """
    k = float(rls.theta[0])
    if not np.isfinite(k) or not (K_MIN_VALID <= k <= K_MAX_VALID):
      return None
    return k

  def model(self) -> HondaSteeringModel:
    # theta is [K, -K*offset, -K*friction, -K*asym], so every derived term is divided
    # back out by the gain that scaled it
    k = self._factor_from(self.steady_rls)
    scale = k if k is not None else self.prior.lat_accel_factor(20.0)
    m = HondaSteeringModel(
      fingerprint=self.fingerprint,
      friction=float(np.clip(-self.steady_rls.theta[2] / scale, 0.0, 0.4)),
      # wider than it was: a skipped roll compensation leaves its crown here, and the
      # old +-0.3 was already railing on route 729a2e65b1f6201d with the compensation on
      offset=float(np.clip(-self.steady_rls.theta[1] / scale, -0.5, 0.5)),
      asymmetry=float(np.clip(-self.steady_rls.theta[3] / scale, -0.5, 0.5)),
      response_tau=float(np.clip(-self.lag_rls.theta[0], 0.0, 0.6)),
      actuator_delay=self.delay_bank.delay,
      delay_learned=self.delay_bank.learned,
      delay_railed=self.delay_bank.railed,
      steer_ratio=self.steer_ratio,
      understeer_gradient=self.understeer_gradient,
      driver_torque_threshold=self.override_est.threshold,
      points=self.points,
      resets=self.resets,
      diverged=k is None,
      asymmetry_learned=self._asym_enabled,
      roll_comp_fraction=self._roll_comp_fraction(),
      lat_accel_torque_corr=self.corr_compensated.value,
      lat_accel_torque_corr_raw=self.corr_raw.value,
    )

    # speed schedule: use a bucket's own fit once it has seen enough of the lat accel
    # range, otherwise fall back to the global fit so the schedule stays monotonic-ish
    bps, vs, learned_buckets = [], [], 0
    prior_gain = self.prior.lat_accel_factor(20.0)
    for i, centre in enumerate(speed_bucket_centres()):
      bucket = self._bucket_gain(i)
      if bucket is not None:
        vs.append(bucket)
        learned_buckets += 1
      else:
        vs.append(k if k is not None else prior_gain)
      bps.append(centre)
    m.lat_accel_factor_bp = bps
    m.lat_accel_factor_v = vs
    m.learned_buckets = learned_buckets

    m.max_useful_torque = self.prior.max_useful_torque

    m.bucket_counts = self.bucket_counts.tolist()
    m.covariance = {
      "steady": self.steady_rls.P.tolist(),
      "speed": [rls.P.tolist() for rls in self.speed_rls],
      "lag": self.lag_rls.P.tolist(),
    }

    m.valid = (k is not None and self.points >= MIN_TOTAL_POINTS and learned_buckets >= 1
               and self._excited(self.bucket_counts.sum(axis=0)))
    return m

  def _roll_comp_fraction(self) -> float:
    seen = self.roll_comp_applied + self.roll_comp_skipped
    return float(self.roll_comp_applied / seen) if seen else 0.0

  def _bucket_gain(self, i: int) -> float | None:
    """A speed bucket's own gain, or None when its coverage does not support one."""
    cells = self.bucket_counts[i]
    if (cells >= MIN_POINTS_PER_BUCKET).sum() < 2 or not self._excited(cells):
      return None
    return self._factor_from(self.speed_rls[i])

  @staticmethod
  def _excited(cells) -> bool:
    """Whether the covered cells span enough lateral acceleration to fit anything.

    Point count is not coverage. Route 729a2e65b1f6201d had 1150 points in one cell and
    100 in its neighbour, which passed a count test easily and still described a band far
    too narrow to separate gain from offset once measurement noise is accounted for.
    """
    covered = np.where(np.asarray(cells) >= MIN_POINTS_PER_BUCKET)[0]
    if len(covered) < 2:
      return False
    span = LAT_ACCEL_BUCKET_EDGES[covered[-1] + 1] - LAT_ACCEL_BUCKET_EDGES[covered[0]]
    return span >= MIN_LAT_ACCEL_SPAN
