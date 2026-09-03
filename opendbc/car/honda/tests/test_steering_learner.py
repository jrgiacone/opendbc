import math

import numpy as np
import pytest

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

# The learner is parameterised by its step time; the fleet sweep runs at 50 Hz, which
# halves what is a very long test and exercises that dt independence. test_learns_at_100hz
# covers the rate Honda control actually runs at.
DT = 0.02
ALL_CARS = list(CAR)


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


@pytest.mark.parametrize("car", ALL_CARS)
def test_prior_is_sane_for_every_platform(car):
  """Every Honda gets a distinct, bounded prior straight from its own CarParams."""
  CP = CarInterface.get_non_essential_params(car)
  prior = prior_from_car_params(CP)
  assert prior.fingerprint == str(car)
  assert 0.6 <= prior.lat_accel_factor(20.0) <= 6.0
  assert 0.03 <= prior.actuator_delay <= 0.45
  assert prior.steer_ratio == pytest.approx(CP.steerRatio)
  assert not prior.valid


def test_priors_differentiate_platforms():
  """The prior must actually separate the fleet, not collapse to one number."""
  factors = {c: prior_from_car_params(CarInterface.get_non_essential_params(c)).lat_accel_factor(20.)
             for c in ALL_CARS}
  assert len(set(round(f, 3) for f in factors.values())) > len(ALL_CARS) // 2
  # the lightest car in the fleet should need the least torque per m/s^2, the heaviest
  # the most
  specs = {c: CarInterface.get_non_essential_params(c).mass for c in ALL_CARS}
  heaviest = sorted(specs, key=specs.get)[-3:]
  assert max(factors, key=factors.get) == min(specs, key=specs.get)
  assert min(factors, key=factors.get) in heaviest


@pytest.mark.parametrize("car", ALL_CARS)
def test_learns_gain_and_friction(car):
  """The learner recovers each car's true EPS gain from a plant it was not primed on."""
  CP = CarInterface.get_non_essential_params(car)
  truth = HondaPlantTruth.for_car(CP)
  ctrl, _ = drive(CP, truth)
  m = ctrl.learner.model()

  assert m.valid, f"{car}: never converged ({m.points} points)"
  assert m.points > 1000
  learned = m.lat_accel_factor(20.0)
  assert learned == pytest.approx(truth.lat_accel_factor, rel=0.20), \
    f"{car}: learned {learned:.3f} vs truth {truth.lat_accel_factor:.3f}"
  assert abs(m.friction - truth.friction) < 0.05
  assert abs(m.offset - truth.offset) < 0.04


@pytest.mark.parametrize("car", ALL_CARS[::7])
def test_learned_beats_prior(car):
  """Learning must improve tracking, not just produce numbers.

  A prior drawn close to the truth cannot be beaten, so the bar depends on how wrong the
  prior actually is for this car: pay for itself where the prior is off, do no harm where
  it is not.
  """
  CP = CarInterface.get_non_essential_params(car)
  truth = HondaPlantTruth.for_car(CP)
  trained, _ = drive(CP, truth)
  learned = trained.learner.model()
  assert learned.valid

  def rms(model):
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
    return float(np.sqrt(np.mean(np.square(errs))))

  learned_rms, prior_rms = rms(learned), rms(None)
  prior_error = abs(trained.prior.lat_accel_factor(22.0) / truth.lat_accel_factor - 1.0)
  if prior_error > 0.15:
    # the prior is materially wrong about this car, so measuring must pay for itself
    assert learned_rms < prior_rms * 0.95, f"{car}: {learned_rms:.4f} vs prior {prior_rms:.4f}"
  else:
    # the prior happens to be nearly right; learning must at least not undo that
    assert learned_rms < prior_rms * 1.25, f"{car}: {learned_rms:.4f} vs prior {prior_rms:.4f}"


@pytest.mark.parametrize("car", ALL_CARS[::5])
def test_learns_command_to_motion_lag(car):
  """Dead time plus rack time constant, the quantity control actually cares about.

  The split between the two is only weakly observable from smooth steering, so the model
  documents and this test checks their sum.
  """
  CP = CarInterface.get_non_essential_params(car)
  truth = HondaPlantTruth.for_car(CP)
  ctrl, _ = drive(CP, truth)
  m = ctrl.learner.model()
  assert abs(m.effective_lag - (truth.delay + truth.tau)) < 0.10, \
    f"{car}: learned lag {m.effective_lag:.3f} vs truth {truth.delay + truth.tau:.3f}"
  assert MIN_DELAY <= m.actuator_delay <= MAX_DELAY


def test_learns_at_100hz():
  """The rate Honda control actually runs at, on a car whose prior is well off."""
  CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
  truth = HondaPlantTruth.for_car(CP)
  ctrl, _ = drive(CP, truth, seconds=250.0, dt=0.01)
  m = ctrl.learner.model()
  assert m.valid
  assert m.lat_accel_factor(20.0) == pytest.approx(truth.lat_accel_factor, rel=0.20)
  assert abs(m.effective_lag - (truth.delay + truth.tau)) < 0.10


def test_learns_speed_schedule():
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


def test_learns_driver_override_threshold():
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


def test_saturated_commands_are_not_learned_from():
  """A rack that gives up at 70% of STEER_MAX must not drag the gain estimate with it."""
  CP = CarInterface.get_non_essential_params(CAR.HONDA_PILOT)
  truth = HondaPlantTruth.for_car(CP)
  truth.max_torque = 0.70
  # demand far more than the rack can give, so the saturated region is actually visited
  ctrl, _ = drive(CP, truth, seconds=500.0, amplitude_scale=2.5)
  m = ctrl.learner.model()
  assert m.valid
  assert m.lat_accel_factor(20.0) == pytest.approx(truth.lat_accel_factor, rel=0.25)


def test_no_learning_while_disengaged_or_overridden():
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


def test_controller_is_inert_and_safe_when_inactive():
  CP = CarInterface.get_non_essential_params(CAR.HONDA_CRV_5G)
  ctrl = HondaAdaptiveLatController(CP, dt=DT)
  for _ in range(100):
    out, dbg = ctrl.update(2.0, 0.0, 25.0, 0.0, 0.0, lat_active=False)
    assert out == 0.0 and ctrl.i == 0.0


@pytest.mark.parametrize("car", ALL_CARS[::4])
def test_output_always_bounded(car):
  """Whatever the model says, the command stays inside the platform's torque range."""
  CP = CarInterface.get_non_essential_params(car)
  ctrl = HondaAdaptiveLatController(CP, dt=DT)
  for i in range(2000):
    out, _ = ctrl.update(50.0 * (-1) ** i, -50.0 * (-1) ** i, 30.0, 0.0, 0.0, lat_active=True)
    assert -1.0 <= out <= 1.0
  assert abs(ctrl.i) <= 0.36


class TestResume:
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
    assert after.lat_accel_factor_v == pytest.approx(before.lat_accel_factor_v, rel=0.02)
    assert after.learned_buckets == before.learned_buckets

  def test_keeps_confidence(self):
    """Resumed, but not at full confidence: the car can change between ignitions."""
    CP, _, model = self.trained()
    resumed = HondaSteeringLearner(CP, dt=DT, learned=model)
    fresh = HondaSteeringLearner(CP, dt=DT)

    trained_var = model.covariance["steady"][0][0]
    assert resumed.steady_rls.P[0][0] == pytest.approx(trained_var * 2.0)
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
    assert second.lat_accel_factor(20.0) == pytest.approx(truth.lat_accel_factor, rel=0.20)

  def test_resumes_the_delay_rather_than_re_racing_it(self):
    CP, _, model = self.trained()
    resumed = HondaSteeringLearner(CP, dt=DT, learned=model)
    assert resumed.model().actuator_delay == pytest.approx(model.actuator_delay)

  @pytest.mark.parametrize("damage", [
    {"bucket_counts": [[1, 2], [3, 4]]},
    {"bucket_counts": "not a list"},
    {"covariance": {"steady": [[1.0]]}},
    {"covariance": {"steady": "nope"}},
    {"covariance": {"steady": [[float("nan")] * 4] * 4}},
    {"bucket_counts": [], "covariance": {}},
  ])
  def test_ignores_a_misshapen_cache(self, damage):
    """A corrupt cache must cost us the resume, not the learner."""
    CP, _, model = self.trained()
    for k, v in damage.items():
      setattr(model, k, v)
    learner = HondaSteeringLearner(CP, dt=DT, learned=model)
    if "covariance" in damage:
      # nothing restored: the fit is as open as an unlearned one seeded with these values
      unrestored = 0.5 * model.lat_accel_factor(20.0) ** 2
      assert learner.steady_rls.P[0][0] == pytest.approx(unrestored)
    # whatever was wrong, the gain still resumes and the learner still runs
    assert learner.steady_rls.theta[0] == pytest.approx(model.lat_accel_factor(20.0))
    learner.update(HondaSteerSample(t=0.0, v_ego=25.0, torque_cmd=0.3, steering_angle_deg=2.0,
                                    steering_rate_deg=1.0, lat_active=True, lat_accel=0.8))


class TestNoisyMeasurement:
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
      centre = (-0.55 if (t < seconds * 0.8 or not flip) else 0.55) if one_signed else 0.0
      amp = 0.08 if one_signed else 1.6
      a = centre + amp * math.sin(2 * math.pi * t / 9.0)
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

  @pytest.mark.parametrize("noise,hold_hz", [(0.0, None), (0.3, None), (0.5, 7), (0.8, 7)])
  def test_noise_does_not_inflate_the_gain(self, noise, hold_hz):
    """The exact conditions of the route: a narrow one-signed band and a stale yaw rate."""
    m = self.feed(noise=noise, hold_hz=hold_hz).model()
    assert not m.diverged
    assert m.lat_accel_factor(22.0) == pytest.approx(2.4, rel=0.15), \
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
    assert m.lat_accel_factor(22.0) == pytest.approx(learner.prior.lat_accel_factor(22.0))

  def test_divergence_resets_rather_than_persisting(self):
    learner = self.feed(noise=0.0, one_signed=False)
    before = learner.resets
    learner.steady_rls.theta[0] = -0.5
    learner._check_divergence()
    assert learner.resets == before + 1
    assert learner.steady_rls.theta[0] == pytest.approx(learner.prior.lat_accel_factor(20.0))

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


class TestCovarianceWindup:
  def test_no_direction_winds_up_without_data(self):
    """An unexcited direction must not grow until the first sample that touches it lands
    with enormous gain."""
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
    learner = HondaSteeringLearner(CP, dt=DT)
    rls = learner.steady_rls
    for _ in range(50000):
      rls.update(np.array([1.0, 1.0, 0.0, 0.0]), 2.4)     # nothing excites columns 2, 3
    assert np.all(np.diag(rls.P) <= rls.p_max + 1e-9)


def test_model_round_trips():
  CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
  m = prior_from_car_params(CP)
  m.valid = True
  assert HondaSteeringModel.from_json(m.to_json()) == m
  with pytest.raises(ValueError):
    HondaSteeringModel.from_dict({"version": 999})
