import math
import unittest

import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.lat_controller import HondaAdaptiveLatController
from opendbc.car.honda.steering_learner import (
  MAX_DELAY,
  MIN_DELAY,
  MIN_TOTAL_POINTS,
  HondaSteeringLearner,
  HondaSteeringModel,
  HondaSteerSample,
  prior_from_car_params,
)
from opendbc.car.honda.tests.honda_steer_plant import HondaPlant, HondaPlantTruth
from opendbc.car.honda.values import CAR

# The learner is parameterized by its step time; the fleet sweep runs at 50 Hz, which
# halves what is a very long test and exercises that dt independence. test_learns_at_100hz
# covers the rate Honda control actually runs at.
DT = 0.02
ALL_CARS = list(CAR)


class approx:
  """Tolerant float (or float sequence) comparison, for `assert x == approx(y)`.

  Lowercase deliberately: it reads as a value at the call site, not as a class.

  opendbc runs its tests under ``unittest-parallel`` and has no pytest dependency, so
  ``pytest.approx`` is not available here. This covers the part of its behavior these
  tests use: the same default tolerances (rel=1e-6, abs=1e-12), an ``abs``-only mode, and
  element-wise comparison of a sequence.
  """

  def __init__(self, expected, rel=1e-6, abs=1e-12):  # noqa: A002 - mirrors pytest.approx
    self.expected = expected
    self.rel = rel
    self.abs = abs

  def _close(self, actual, expected):
    return abs(actual - expected) <= max(self.rel * abs(expected), self.abs)

  def __eq__(self, actual):
    if isinstance(self.expected, list | tuple):
      return (len(actual) == len(self.expected)
              and all(self._close(a, e) for a, e in zip(actual, self.expected, strict=True)))
    return self._close(actual, self.expected)

  def __repr__(self):
    return f"approx({self.expected!r}, rel={self.rel}, abs={self.abs})"


def drive(CP, truth, seconds=250.0, learned=None, v_profile=None, amplitude_scale=1.0, dt=None):
  """Run the adaptive controller against the plant and return (controller, plant)."""
  dt = DT if dt is None else dt
  plant = HondaPlant(truth, dt)
  ctrl = HondaAdaptiveLatController(CP, dt=dt, learned=learned)
  n = int(seconds / dt)
  rng = np.random.default_rng(0)

  for i in range(n):
    t = i * dt
    # a mix of speeds and curvatures: highway sweepers, suburban curves, and lane changes
    v = v_profile(t) if v_profile else 12.0 + 12.0 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 240.0))
    # Scale the demand to the car's authority so the command stays mostly inside what the
    # rack can deliver - a saturated command teaches nothing, on the road as here - but
    # never below what ordinary driving asks for: a real Odyssey takes the same curves as
    # a real Civic, and too gentle a demand starves the lag fit of anything to measure.
    amp = amplitude_scale * float(np.clip(0.45 * truth.lat_accel_factor, 0.8, 2.2))
    desired = amp * (0.55 * math.sin(2 * math.pi * t / 17.0)
                     + 0.32 * math.sin(2 * math.pi * t / 6.3 + 1.0)
                     + 0.13 * math.sin(2 * math.pi * t / 2.1))
    jerk = amp * (0.55 * 2 * math.pi / 17.0 * math.cos(2 * math.pi * t / 17.0)
                  + 0.32 * 2 * math.pi / 6.3 * math.cos(2 * math.pi * t / 6.3 + 1.0)
                  + 0.13 * 2 * math.pi / 2.1 * math.cos(2 * math.pi * t / 2.1))
    measured = plant.lat_accel + rng.normal(0.0, 0.01)

    torque, _ = ctrl.update(desired, measured, v, plant.angle_deg, plant.rate_deg,
                            lat_active=True, driver_torque=rng.normal(0.0, 120.0),
                            yaw_rate=plant.lat_accel / max(v, 1.0), desired_lat_jerk=jerk)
    plant.step(torque, v)
  return ctrl, plant


class TestFleet(unittest.TestCase):
  """The whole Honda/Acura fleet, swept platform by platform."""

  def test_prior_is_sane_for_every_platform(self):
    """Every Honda gets a distinct, bounded prior straight from its own CarParams."""
    for car in ALL_CARS:
      with self.subTest(car=car):
        CP = CarInterface.get_non_essential_params(car)
        prior = prior_from_car_params(CP)
        assert prior.fingerprint == str(car)
        assert 0.6 <= prior.lat_accel_factor(20.0) <= 6.0
        assert 0.03 <= prior.actuator_delay <= 0.45
        assert prior.steer_ratio == approx(CP.steerRatio)
        assert not prior.valid

  def test_priors_differentiate_platforms(self):
    """The prior must actually separate the fleet, not collapse to one number."""
    factors = {c: prior_from_car_params(CarInterface.get_non_essential_params(c)).lat_accel_factor(20.)
               for c in ALL_CARS}
    assert len({round(f, 3) for f in factors.values()}) > len(ALL_CARS) // 2
    # the lightest car in the fleet should need the least torque per m/s^2, the heaviest
    # the most
    specs = {c: CarInterface.get_non_essential_params(c).mass for c in ALL_CARS}
    # __getitem__ rather than .get as the sort key: .get is typed as returning
    # `float | None`, which is not comparable, so sorted/max/min have no matching overload
    heaviest = sorted(specs, key=specs.__getitem__)[-3:]
    assert max(factors, key=factors.__getitem__) == min(specs, key=specs.__getitem__)
    assert min(factors, key=factors.__getitem__) in heaviest

  def test_learns_gain_and_friction(self):
    """The learner recovers each car's true EPS gain from a plant it was not primed on."""
    for car in ALL_CARS:
      with self.subTest(car=car):
        CP = CarInterface.get_non_essential_params(car)
        truth = HondaPlantTruth.for_car(CP)
        ctrl, _ = drive(CP, truth)
        m = ctrl.learner.model()

        assert m.valid, f"{car}: never converged ({m.points} points)"
        assert m.points > 1000
        learned = m.lat_accel_factor(20.0)
        assert learned == approx(truth.lat_accel_factor, rel=0.20), \
          f"{car}: learned {learned:.3f} vs truth {truth.lat_accel_factor:.3f}"
        assert abs(m.friction - truth.friction) < 0.05
        assert abs(m.offset - truth.offset) < 0.04

  def test_learned_beats_prior(self):
    """Learning must improve tracking, not just produce numbers.

    A prior drawn close to the truth cannot be beaten, so the bar depends on how wrong the
    prior actually is for this car: pay for itself where the prior is off, do no harm where
    it is not.
    """
    for car in ALL_CARS[::7]:
      with self.subTest(car=car):
        CP = CarInterface.get_non_essential_params(car)
        truth = HondaPlantTruth.for_car(CP)
        trained, _ = drive(CP, truth)
        learned = trained.learner.model()
        assert learned.valid

        def rms(model, CP=CP, truth=truth):
          ctrl = HondaAdaptiveLatController(CP, dt=DT, learned=model)
          plant = HondaPlant(truth, DT)
          errs = []
          for i in range(int(120.0 / DT)):
            t = i * DT
            v = 22.0
            desired = 1.5 * math.sin(2 * math.pi * t / 9.0)
            jerk = 1.5 * 2 * math.pi / 9.0 * math.cos(2 * math.pi * t / 9.0)
            torque, _ = ctrl.update(desired, plant.lat_accel, v, plant.angle_deg, plant.rate_deg,
                                    lat_active=True, desired_lat_jerk=jerk)
            plant.step(torque, v)
            if t > 20.0:
              errs.append(desired - plant.lat_accel)
          return float(np.sqrt(np.square(errs).sum() / len(errs)))

        learned_rms, prior_rms = rms(learned), rms(None)
        prior_error = abs(trained.prior.lat_accel_factor(22.0) / truth.lat_accel_factor - 1.0)
        if prior_error > 0.15:
          # the prior is materially wrong about this car, so measuring must pay for itself
          assert learned_rms < prior_rms * 0.95, f"{car}: {learned_rms:.4f} vs prior {prior_rms:.4f}"
        else:
          # the prior happens to be nearly right; learning must at least not undo that
          assert learned_rms < prior_rms * 1.25, f"{car}: {learned_rms:.4f} vs prior {prior_rms:.4f}"

  def test_learns_command_to_motion_lag(self):
    """Dead time plus rack time constant, the quantity control actually cares about.

    The split between the two is only weakly observable from smooth steering, so the model
    documents and this test checks their sum.
    """
    for car in ALL_CARS[::5]:
      with self.subTest(car=car):
        CP = CarInterface.get_non_essential_params(car)
        truth = HondaPlantTruth.for_car(CP)
        ctrl, _ = drive(CP, truth)
        m = ctrl.learner.model()
        assert abs(m.effective_lag - (truth.delay + truth.tau)) < 0.10, \
          f"{car}: learned lag {m.effective_lag:.3f} vs truth {truth.delay + truth.tau:.3f}"
        assert MIN_DELAY <= m.actuator_delay <= MAX_DELAY

  def test_output_always_bounded(self):
    """Whatever the model says, the command stays inside the platform's torque range."""
    for car in ALL_CARS[::4]:
      with self.subTest(car=car):
        CP = CarInterface.get_non_essential_params(car)
        ctrl = HondaAdaptiveLatController(CP, dt=DT)
        for i in range(2000):
          out, _ = ctrl.update(50.0 * (-1) ** i, -50.0 * (-1) ** i, 30.0, 0.0, 0.0, lat_active=True)
          assert -1.0 <= out <= 1.0
        assert abs(ctrl.i) <= 0.36


class TestLearning(unittest.TestCase):
  def test_learns_at_100hz(self):
    """The rate Honda control actually runs at, on a car whose prior is well off."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    truth = HondaPlantTruth.for_car(CP)
    ctrl, _ = drive(CP, truth, seconds=250.0, dt=0.01)
    m = ctrl.learner.model()
    assert m.valid
    assert m.lat_accel_factor(20.0) == approx(truth.lat_accel_factor, rel=0.20)
    assert abs(m.effective_lag - (truth.delay + truth.tau)) < 0.10

  def test_learns_speed_schedule(self):
    """A car whose gain varies with speed gets a schedule, not one number."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_BOSCH)
    truth = HondaPlantTruth.for_car(CP)

    learner = HondaSteeringLearner(CP, dt=DT)
    rng = np.random.default_rng(1)
    for i in range(int(1200.0 / DT)):
      t = i * DT
      v = 10.0 + 20.0 * (i % 30000) / 30000.0
      # true gain falls off with speed, as Honda EPS assist does
      k = truth.lat_accel_factor * (1.0 - 0.010 * (v - 10.0))
      a = 1.5 * math.sin(2 * math.pi * t / 8.0) + 0.4 * rng.normal()
      u = a / k + truth.offset
      learner.update(HondaSteerSample(t=t, v_ego=v, torque_cmd=u, steering_angle_deg=0.0,
                                      steering_rate_deg=math.degrees(a), lat_active=True,
                                      lat_accel=a))
    m = learner.model()
    assert m.valid and m.learned_buckets >= 2
    assert m.lat_accel_factor(10.0) > m.lat_accel_factor(30.0) * 1.05

  def test_learns_driver_override_threshold(self):
    CP = CarInterface.get_non_essential_params(CAR.ACURA_RDX)
    learner = HondaSteeringLearner(CP, dt=DT)
    rng = np.random.default_rng(2)
    for i in range(20000):
      learner.update(HondaSteerSample(t=i * DT, v_ego=25.0, torque_cmd=0.0,
                                      steering_angle_deg=0.0, steering_rate_deg=0.0,
                                      driver_torque=rng.normal(0.0, 40.0), lat_active=True,
                                      lat_accel=0.0))
    th = learner.model().driver_torque_threshold
    assert 300.0 <= th <= 700.0, th

  def test_saturated_commands_are_not_learned_from(self):
    """A rack that gives up at 70% of STEER_MAX must not drag the gain estimate with it."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_PILOT)
    truth = HondaPlantTruth.for_car(CP)
    truth.max_torque = 0.70
    # demand far more than the rack can give, so the saturated region is actually visited
    ctrl, _ = drive(CP, truth, seconds=500.0, amplitude_scale=2.5)
    m = ctrl.learner.model()
    assert m.valid
    assert m.lat_accel_factor(20.0) == approx(truth.lat_accel_factor, rel=0.25)

  def test_no_learning_while_disengaged_or_overridden(self):
    CP = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
    learner = HondaSteeringLearner(CP, dt=DT)
    for i in range(5000):
      learner.update(HondaSteerSample(t=i * DT, v_ego=25.0, torque_cmd=0.5,
                                      steering_angle_deg=5.0, steering_rate_deg=1.0,
                                      lat_active=False, lat_accel=1.0))
      learner.update(HondaSteerSample(t=i * DT, v_ego=25.0, torque_cmd=0.5,
                                      steering_angle_deg=5.0, steering_rate_deg=1.0,
                                      lat_active=True, steering_pressed=True, lat_accel=1.0))
    assert learner.points == 0
    assert not learner.model().valid

  def test_controller_is_inert_and_safe_when_inactive(self):
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
    ctrl = HondaAdaptiveLatController(CP, dt=DT)
    for _ in range(100):
      out, _dbg = ctrl.update(2.0, 0.0, 25.0, 0.0, 0.0, lat_active=False)
      assert out == 0.0 and ctrl.i == 0.0


class TestResume(unittest.TestCase):
  """A restart must continue a model, not merely copy its numbers."""

  @staticmethod
  def trained(car=CAR.HONDA_CIVIC_2022):
    CP = CarInterface.get_non_essential_params(car)
    truth = HondaPlantTruth.for_car(CP)
    ctrl, _ = drive(CP, truth)
    model = ctrl.learner.model()
    assert model.valid
    return CP, truth, model

  def test_keeps_the_speed_schedule(self):
    """Seeding every bucket from one point on the schedule would flatten it each ignition."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_BOSCH)
    learner = HondaSteeringLearner(CP, dt=DT)
    rng = np.random.default_rng(1)
    for i in range(int(1200.0 / DT)):
      t = i * DT
      v = 10.0 + 20.0 * ((i * DT) % 300.0) / 300.0
      k = 2.5 * (1.0 - 0.015 * (v - 10.0))
      a = 1.5 * math.sin(2 * math.pi * t / 8.0) + 0.4 * rng.normal()
      learner.update(HondaSteerSample(t=t, v_ego=v, torque_cmd=a / k, steering_angle_deg=0.0,
                                      steering_rate_deg=math.degrees(a), lat_active=True,
                                      lat_accel=a))
    before = learner.model()
    assert before.learned_buckets >= 2
    spread = max(before.lat_accel_factor_v) - min(before.lat_accel_factor_v)
    assert spread > 0.1, before.lat_accel_factor_v

    after = HondaSteeringLearner(CP, dt=DT, learned=before).model()
    assert after.lat_accel_factor_v == approx(before.lat_accel_factor_v, rel=0.02)
    assert after.learned_buckets == before.learned_buckets

  def test_keeps_confidence(self):
    """Resumed, but not at full confidence: the car can change between ignitions."""
    CP, _, model = self.trained()
    resumed = HondaSteeringLearner(CP, dt=DT, learned=model)
    fresh = HondaSteeringLearner(CP, dt=DT)

    trained_var = model.covariance["steady"][0][0]
    assert resumed.steady_rls.P[0][0] == approx(trained_var * 2.0)
    assert resumed.steady_rls.P[0][0] < fresh.steady_rls.P[0][0]

  def test_keeps_lifetime_points_and_stays_converged(self):
    CP, _, model = self.trained()
    resumed = HondaSteeringLearner(CP, dt=DT, learned=model).model()
    assert resumed.points == model.points
    assert resumed.valid, "a resumed model must not report itself unconverged"

  def test_accumulates_across_drives(self):
    """Two short drives must leave a model built on both, not on the second alone."""
    CP, truth, first = self.trained()
    ctrl = HondaAdaptiveLatController(CP, dt=DT, learned=first)
    plant = HondaPlant(truth, DT)
    for i in range(int(120.0 / DT)):
      t = i * DT
      desired = 1.2 * math.sin(2 * math.pi * t / 11.0)
      jerk = 1.2 * 2 * math.pi / 11.0 * math.cos(2 * math.pi * t / 11.0)
      torque, _ = ctrl.update(desired, plant.lat_accel, 24.0, plant.angle_deg, plant.rate_deg,
                              lat_active=True, desired_lat_jerk=jerk)
      plant.step(torque, 24.0)
    second = ctrl.learner.model()
    assert second.points > first.points
    assert second.lat_accel_factor(20.0) == approx(truth.lat_accel_factor, rel=0.20)

  def test_resumes_the_delay_rather_than_re_racing_it(self):
    CP, _, model = self.trained()
    resumed = HondaSteeringLearner(CP, dt=DT, learned=model)
    assert resumed.model().actuator_delay == approx(model.actuator_delay)

  def test_ignores_a_misshapen_cache(self):
    """A corrupt cache must cost us the resume, not the learner."""
    for damage in [
      {"bucket_counts": [[1, 2], [3, 4]]},
      {"bucket_counts": "not a list"},
      {"covariance": {"steady": [[1.0]]}},
      {"covariance": {"steady": "nope"}},
      {"covariance": {"steady": [[float("nan")] * 4] * 4}},
      {"bucket_counts": [], "covariance": {}},
    ]:
      with self.subTest(damage=damage):
        CP, _, model = self.trained()
        for k, v in damage.items():
          setattr(model, k, v)
        learner = HondaSteeringLearner(CP, dt=DT, learned=model)
        if "covariance" in damage:
          # nothing restored: the fit is as open as an unlearned one seeded with these values
          unrestored = 0.5 * model.lat_accel_factor(20.0) ** 2
          assert learner.steady_rls.P[0][0] == approx(unrestored)
        # whatever was wrong, the gain still resumes and the learner still runs
        assert learner.steady_rls.theta[0] == approx(model.lat_accel_factor(20.0))
        learner.update(HondaSteerSample(t=0.0, v_ego=25.0, torque_cmd=0.3, steering_angle_deg=2.0,
                                        steering_rate_deg=1.0, lat_active=True, lat_accel=0.8))


class TestNoisyMeasurement(unittest.TestCase):
  """Regression tests for route 729a2e65b1f6201d, where the published gain railed at 8.0.

  The fit is oriented with the command as the regressor and the measured lateral
  acceleration as the dependent variable. Oriented the other way, noise on the
  measurement biased its own coefficient (1/K) toward zero and the published gain, being
  its reciprocal, ran away upward.
  """

  @staticmethod
  def feed(noise=0.0, hold_hz=None, one_signed=True, seconds=185.0, seed=0, truth_k=2.4,
           flip=True):
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    learner = HondaSteeringLearner(CP, dt=DT)
    friction, offset, tau = 0.05, 0.02, 0.15
    rng = np.random.default_rng(seed)
    t, held, last_hold = 0.0, 0.0, -1e9
    while t < seconds:
      t += DT
      center = (-0.55 if (t < seconds * 0.8 or not flip) else 0.55) if one_signed else 0.0
      amp = 0.08 if one_signed else 1.6
      a = center + amp * math.sin(2 * math.pi * t / 9.0)
      a_dot = amp * 2 * math.pi / 9.0 * math.cos(2 * math.pi * t / 9.0)
      u = a / truth_k + tau * a_dot / truth_k + offset + friction * np.sign(a_dot)
      measured = a + rng.normal(0.0, noise)
      if hold_hz is not None:            # yaw rate held between slow deviceMotion updates
        if t - last_hold >= 1.0 / hold_hz:
          held, last_hold = measured, t
        measured = held
      learner.update(HondaSteerSample(t=t, v_ego=22.0, torque_cmd=u, steering_angle_deg=a * 3,
                                      steering_rate_deg=math.degrees(a_dot) * 5,
                                      lat_active=True, lat_accel=measured))
    return learner

  def test_noise_does_not_inflate_the_gain(self):
    """The exact conditions of the route: a narrow one-signed band and a stale yaw rate."""
    for noise, hold_hz in [(0.0, None), (0.3, None), (0.5, 7), (0.8, 7)]:
      with self.subTest(noise=noise, hold_hz=hold_hz):
        m = self.feed(noise=noise, hold_hz=hold_hz).model()
        assert not m.diverged
        assert m.lat_accel_factor(22.0) == approx(2.4, rel=0.15), \
          f"noise={noise} hold={hold_hz}: gain {m.lat_accel_factor(22.0):.2f}"

  def test_a_diverged_fit_is_surfaced_not_masked(self):
    """A diverged fit must not be indistinguishable from an unfitted one."""
    learner = self.feed(noise=0.0, one_signed=False)
    assert learner.model().valid
    # force every fit outside the physically possible range
    for rls in [learner.steady_rls] + learner.speed_rls:
      rls.theta[0] = -0.5
    m = learner.model()
    assert m.diverged
    assert not m.valid, "a diverged fit must never be published as converged"
    # with nothing measured, the published gain is the prior - but flagged, not silent
    assert m.lat_accel_factor(22.0) == approx(learner.prior.lat_accel_factor(22.0))

  def test_a_railed_secondary_term_is_surfaced_not_masked(self):
    """friction/offset/asymmetry are clipped before publishing same as the gain is - a fit
    that has genuinely left the physically possible range is a diverged fit too, even with
    the gain itself still in range.

    Route 729a2e65b1f6201d/00000011 published asymmetry pinned at the +0.5 publish-clip
    rail for the back third of a 40 minute drive with ``diverged`` reporting False
    throughout, because only the gain fed it.
    """
    learner = self.feed(noise=0.0, one_signed=False)
    assert learner.model().valid
    assert not learner.model().diverged
    # push the asymmetry term well past ASYMMETRY_MAX_VALID; gain is untouched
    learner.steady_rls.theta[3] = -10.0 * learner.steady_rls.theta[0]
    m = learner.model()
    assert m.asymmetry == approx(0.5), "still clipped for publishing"
    assert m.diverged, "a fit railed past ASYMMETRY_MAX_VALID must be flagged, not silently clipped"
    assert not m.valid, "a diverged fit must never be published as converged"

  def test_ordinary_fit_noise_does_not_trip_divergence(self):
    """A term merely brushing its publish-clip boundary is not the same as one that has
    left the physically possible range - only the wider *_MIN_VALID/*_MAX_VALID bounds
    (mirroring K_MIN_VALID/K_MAX_VALID) should mark a fit diverged."""
    learner = self.feed(noise=0.0, one_signed=False)
    m = learner.model()
    assert not m.diverged
    # sits right at the publish-clip edge, well inside FRICTION_MIN_VALID/MAX_VALID
    learner.steady_rls.theta[2] = -0.45 * learner.steady_rls.theta[0]
    m = learner.model()
    assert m.friction == approx(0.4)
    assert not m.diverged, "brushing the publish-clip bound alone must not read as diverged"

  def test_divergence_resets_rather_than_persisting(self):
    learner = self.feed(noise=0.0, one_signed=False)
    before = learner.resets
    learner.steady_rls.theta[0] = -0.5
    learner._check_divergence()
    assert learner.resets == before + 1
    assert learner.steady_rls.theta[0] == approx(learner.prior.lat_accel_factor(20.0))

  def test_a_railed_secondary_term_resets_only_itself(self):
    """A friction/offset/asymmetry divergence must recover, not stay flagged forever - and
    must not throw away a gain, or the other two terms, that are still sound."""
    learner = self.feed(noise=0.0, one_signed=False)
    k_before = learner.steady_rls.theta[0]
    offset_before = learner.steady_rls.theta[1]
    before = learner.resets
    # push only the asymmetry term past ASYMMETRY_MAX_VALID
    learner.steady_rls.theta[3] = -10.0 * k_before
    learner._check_divergence()
    assert learner.resets == before + 1
    assert learner.steady_rls.theta[3] == 0.0, "the railed column resets to nothing measured"
    assert learner.steady_rls.theta[0] == approx(k_before), "gain is untouched"
    assert learner.steady_rls.theta[1] == approx(offset_before), "offset is untouched"
    assert not learner.model().diverged, "a reset column is no longer railed"

  def test_a_railed_steer_ratio_resets_rather_than_persisting(self):
    """Steer ratio is a separate fit (``sr_rls``) from the torque model, but the same
    failure mode applies: a fit that has left the physically possible range must recover,
    not be clipped and published as if it were a measurement forever."""
    learner = self.feed(noise=0.0, one_signed=False)
    before = learner.resets
    learner.sr_rls.theta[0] = 50.0  # past STEER_RATIO_MAX_VALID
    # one more accepted sample is enough: the check runs on every sr_rls update
    learner.update(HondaSteerSample(t=1e6, v_ego=22.0, torque_cmd=0.1, steering_angle_deg=1.0,
                                    steering_rate_deg=0.0, lat_active=True, lat_accel=0.2))
    assert learner.resets == before + 1
    assert learner.sr_rls.theta[0] == approx(learner.prior.steer_ratio)
    assert learner.steer_ratio == approx(learner.prior.steer_ratio)

  def test_asymmetry_waits_for_both_directions(self):
    """max(u, 0) duplicates the command column while the command is one-signed."""
    one_way = self.feed(one_signed=True, seconds=120.0, flip=False)
    assert not one_way._asym_enabled
    assert one_way.model().asymmetry == 0.0
    assert not one_way.model().asymmetry_learned

    both = self.feed(one_signed=False)
    assert both._asym_enabled and both.model().asymmetry_learned

  def test_a_narrow_band_is_not_treated_as_coverage(self):
    """Point count is not coverage: 1000 points in one cell still fit nothing."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    learner = HondaSteeringLearner(CP, dt=DT)
    for i in range(20000):
      t = i * DT
      a = -0.55 + 0.02 * math.sin(2 * math.pi * t / 9.0)   # one cell, forever
      learner.update(HondaSteerSample(t=t, v_ego=22.0, torque_cmd=a / 2.4 + 0.02,
                                      steering_angle_deg=a * 3, steering_rate_deg=0.5,
                                      lat_active=True, lat_accel=a))
    m = learner.model()
    assert m.points > MIN_TOTAL_POINTS, "the point count alone would have passed"
    assert not m.valid, "a single narrow band must not count as a converged model"

  def test_a_stale_measurement_is_not_fitted(self):
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    learner = HondaSteeringLearner(CP, dt=DT)
    for i in range(5000):
      t = i * DT
      a = 1.2 * math.sin(2 * math.pi * t / 9.0)
      learner.update(HondaSteerSample(t=t, v_ego=22.0, torque_cmd=a / 2.4, steering_angle_deg=a * 3,
                                      steering_rate_deg=5.0, lat_active=True, lat_accel=a,
                                      lat_accel_valid=False))
    assert learner.points == 0

  def test_saturated_fraction_is_evidence_not_a_fit_input(self):
    """max_useful_torque is carried from the prior, not learned; saturated_fraction is the
    evidence for whether that is actually limiting this car, without ever changing it."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    learner = HondaSteeringLearner(CP, dt=DT)
    assert learner.model().saturated_fraction == 0.0, "no data yet"

    for i in range(1000):
      t = i * DT
      saturated = i % 4 == 0   # exactly a quarter of samples
      learner.update(HondaSteerSample(t=t, v_ego=22.0, torque_cmd=1.0 if saturated else 0.1,
                                      steering_angle_deg=0.0, steering_rate_deg=0.0,
                                      lat_active=True, saturated=saturated))
    assert learner.model().saturated_fraction == approx(0.25)

    before = learner.model().saturated_fraction
    for i in range(1000):
      # disengaged and pressed samples must not count, saturated or not
      learner.update(HondaSteerSample(t=1000 * DT + i * DT, v_ego=22.0, torque_cmd=1.0,
                                      steering_angle_deg=0.0, steering_rate_deg=0.0,
                                      lat_active=False, saturated=True))
      learner.update(HondaSteerSample(t=1000 * DT + i * DT, v_ego=22.0, torque_cmd=1.0,
                                      steering_angle_deg=0.0, steering_rate_deg=0.0,
                                      lat_active=True, steering_pressed=True, saturated=True))
    assert learner.model().saturated_fraction == approx(before)


class TestCovarianceWindup(unittest.TestCase):
  def test_no_direction_winds_up_without_data(self):
    """An unexcited direction must not grow until the first sample that touches it lands
    with enormous gain."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    learner = HondaSteeringLearner(CP, dt=DT)
    rls = learner.steady_rls
    for _ in range(50000):
      rls.update(np.array([1.0, 1.0, 0.0, 0.0]), 2.4)     # nothing excites columns 2, 3
    assert np.all(np.diag(rls.P) <= rls.p_max + 1e-9)


class TestModelSerialization(unittest.TestCase):
  def test_model_round_trips(self):
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    m = prior_from_car_params(CP)
    m.valid = True
    assert HondaSteeringModel.from_json(m.to_json()) == m
    with self.assertRaises(ValueError):
      HondaSteeringModel.from_dict({"version": 999})


class TestRollCompensationGate(unittest.TestCase):
  """Roll enters the fitted target as sin(roll)*9.81, so a bad estimate is not a small
  error: on route 729a2e65b1f6201d it had a standard deviation 1.07x the lateral
  acceleration signal itself. The gate rejects an estimate that cannot be road crown; the
  published correlations are what decide whether the compensation belongs in the fit."""

  @staticmethod
  def _learner():
    return HondaSteeringLearner(CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022), dt=DT)

  @staticmethod
  def _sample(roll, **kw):
    return HondaSteerSample(t=0.0, v_ego=20.0, torque_cmd=0.2, steering_angle_deg=1.0,
                            steering_rate_deg=0.0, lat_active=True, yaw_rate=0.05,
                            roll=roll, lat_accel_valid=True, **kw)

  def test_a_plausible_roll_is_applied(self):
    learner = self._learner()
    roll = 0.02                                   # ~1.1 deg of crown, 0.20 m/s^2
    settled = learner._lat_accel(self._sample(roll))
    for _ in range(200):                          # let the rate filter settle
      settled = learner._lat_accel(self._sample(roll))
    assert settled == approx(0.05 * 20.0 - math.sin(roll) * 9.81, abs=1e-6)
    assert learner.roll_comp_applied > 0

  def test_an_implausibly_large_roll_is_skipped(self):
    """Superelevation tops out near 10%; anything past that is the estimate, not the road."""
    learner = self._learner()
    roll = 0.5                                    # 4.7 m/s^2, five times any real crown
    a = learner._lat_accel(self._sample(roll))
    assert a == approx(0.05 * 20.0, abs=1e-6), "the compensation must not be applied"
    assert learner.roll_comp_skipped == 1

  def test_a_roll_estimate_that_is_still_settling_is_skipped(self):
    """A localizer converging looks like the road banking at an impossible rate."""
    learner = self._learner()
    skipped_before = learner.roll_comp_skipped
    for i in range(50):                           # 0.6 rad/s, an order of magnitude too fast
      learner._lat_accel(self._sample(i * 0.6 * DT))
    assert learner.roll_comp_skipped > skipped_before

  def test_the_gate_is_reported(self):
    learner = self._learner()
    for _ in range(100):
      learner._lat_accel(self._sample(0.02))
    for _ in range(100):
      learner._lat_accel(self._sample(0.5))
    assert learner.model().roll_comp_fraction == approx(0.5, abs=0.02)

  def test_both_correlations_are_published(self):
    """The compensated and uncompensated targets are scored against the same command, so
    a log says whether removing the roll estimate helped or hurt."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    learner = HondaSteeringLearner(CP, dt=DT)
    gain, v_ego = 2.0, 20.0
    rng = np.random.default_rng(0)
    cmd = 0.0
    for i in range(20000):
      cmd = float(np.clip(cmd + (rng.normal(0.0, 0.25) - cmd) * 0.01, -0.6, 0.6))
      # a pure command response, with a roll estimate that is nothing but noise
      lat_accel = gain * cmd
      roll = float(rng.normal(0.0, 0.01))
      learner.update(HondaSteerSample(
        t=i * DT, v_ego=v_ego, torque_cmd=cmd,
        steering_angle_deg=math.degrees(lat_accel / v_ego ** 2 * learner.steer_ratio * learner.wheelbase),
        steering_rate_deg=0.0, lat_active=True,
        yaw_rate=(lat_accel + math.sin(roll) * 9.81) / v_ego, roll=roll, lat_accel_valid=True,
      ))

    m = learner.model()
    # the roll here is real, so removing it must leave the command tracking the target
    # at least as well as leaving it in
    assert m.lat_accel_torque_corr > 0.5
    assert m.lat_accel_torque_corr >= m.lat_accel_torque_corr_raw


class TestLagIsNotOverclaimed(unittest.TestCase):
  """The dead time is published as a measurement only when it beat the prior.

  Fitting (dead time, tau) by output error over route 729a2e65b1f6201d leaves 0.2%
  residual leverage on tau and 0.6% on the dead time, against 224% and 578% on a
  synthetic rack of known truth at the same noise and excitation. On a surface that flat
  the argmin is a coin toss, so the bank holds the prior unless some candidate is
  meaningfully better than the prior's own. The fit itself is untouched: tau stays free
  to trade against the dead time, which is what lands their sum in the right place.
  """

  def test_a_flat_race_is_not_called_a_measurement(self):
    """No command at all: every candidate explains the data identically.

    The delay is still published - the static fit has to align on something, and holding
    it at the prior contaminates gain, friction and offset instead - but delay_learned
    has to say that nothing was actually learned.
    """
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    learner = HondaSteeringLearner(CP, dt=DT)
    for i in range(20000):
      learner.update(HondaSteerSample(
        t=i * DT, v_ego=20.0, torque_cmd=0.0, steering_angle_deg=0.0,
        steering_rate_deg=0.0, lat_active=True, lat_accel=0.0, lat_accel_valid=True))
    m = learner.model()
    assert not m.delay_learned, "an unwon race must not be published as a measurement"
    assert MIN_DELAY <= m.actuator_delay <= MAX_DELAY

  def test_a_won_race_is_called_a_measurement(self):
    """A plant with a real dead time, driven hard enough to find it."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    truth = HondaPlantTruth.for_car(CP)
    ctrl, _ = drive(CP, truth)
    assert ctrl.learner.model().delay_learned

  def test_a_real_lag_still_clears_the_bar(self):
    """The gate must not block a lag the data genuinely supports.

    This is the case that says the gate is a publication rule and not a lobotomy: on a
    plant with a real dead time and a real time constant, the learned sum still lands.
    """
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    truth = HondaPlantTruth.for_car(CP)
    ctrl, _ = drive(CP, truth)
    m = ctrl.learner.model()
    assert abs(m.effective_lag - (truth.delay + truth.tau)) < 0.10
