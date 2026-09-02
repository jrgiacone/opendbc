import math

import numpy as np
import pytest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.lat_controller import HondaAdaptiveLatController
from opendbc.car.honda.steering_learner import (
  MAX_DELAY,
  MIN_DELAY,
  HondaSteeringLearner,
  HondaSteeringModel,
  HondaSteerSample,
  prior_from_car_params,
)
from opendbc.car.honda.tests.honda_steer_plant import HondaPlant, HondaPlantTruth
from opendbc.car.honda.values import CAR

DT = 0.01
ALL_CARS = list(CAR)


def drive(CP, truth, seconds=350.0, learned=None, v_profile=None):
  """Run the adaptive controller against the plant and return (controller, plant)."""
  plant = HondaPlant(truth, DT)
  ctrl = HondaAdaptiveLatController(CP, dt=DT, learned=learned)
  n = int(seconds / DT)
  rng = np.random.default_rng(0)

  for i in range(n):
    t = i * DT
    # a mix of speeds and curvatures: highway sweepers, suburban curves, and lane changes
    v = v_profile(t) if v_profile else 12.0 + 12.0 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 240.0))
    # keep the demand inside what the rack can actually deliver: a saturated command
    # teaches nothing, which is true on the road as well as here
    amp = float(np.clip(0.45 * truth.lat_accel_factor, 0.4, 2.2))
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
  """Learning must improve tracking, not just produce numbers."""
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

  assert rms(learned) < rms(None) * 0.95


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


def test_learns_deadzone_and_saturation():
  CP = CarInterface.get_non_essential_params(CAR.HONDA_PILOT)
  truth = HondaPlantTruth.for_car(CP)
  truth.deadzone = 0.10
  truth.max_torque = 0.70
  ctrl, _ = drive(CP, truth, seconds=600.0)
  m = ctrl.learner.model()
  assert m.max_useful_torque <= 0.9
  assert m.deadzone <= 0.3


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


def test_model_round_trips():
  CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022)
  m = prior_from_car_params(CP)
  m.valid = True
  assert HondaSteeringModel.from_json(m.to_json()) == m
  with pytest.raises(ValueError):
    HondaSteeringModel.from_dict({"version": 999})
