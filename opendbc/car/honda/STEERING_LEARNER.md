# Honda self-identifying steering

Honda lateral control is currently a table of hand tuned constants. `interface.py` carries
a branch per platform setting `lateralParams.torqueBP/torqueV` and `lateralTuning.pid.kpV/kiV`,
every Honda shares one feedforward gain (`kf = 0.00006`) and one actuator delay (0.1 s),
and `values.py::STEER_THRESHOLD` carries a second per-platform table for driver override.

That has to cover a very wide fleet. The Honda/Acura routes in `opendbc/car/tests/routes.py`
span:

| axis | range across the fleet |
| --- | --- |
| EPS/camera generation | Nidec, Bosch, Bosch radarless, Bosch CAN FD |
| CAN torque scale (`torqueBP[-1]`) | 1000 (CR-V) … 32767 (Odyssey TWN) |
| curb mass | 890 kg (N-Box) … ~2200 kg (Pilot 4G / MDX 4G) |
| steer ratio | ~10.6 … ~21 |
| existing `kpV` | 0.115 … 1.1 |
| markets | US, EU, Brazil, Thailand, Taiwan, Singapore, South Africa, Japan kei |

A constant that is right for a 2016 Civic on a Nidec rack is not right for a 2025 MDX on
Bosch CAN FD, and no amount of table filling fixes an individual car's rack friction,
alignment bias or worn-in asymmetry. This module measures those instead.

## What is identified

`steering_learner.py` fits, per fingerprint and online:

* **`lat_accel_factor(v)`** — m/s² of lateral acceleration per unit of normalized torque
  command, on a speed schedule (Honda assist varies with speed, most visibly on Nidec).
* **`friction`** — command needed just to break the rack loose.
* **`offset`** — steady state bias: road crown, alignment, EPS trim.
* **`asymmetry`** — left/right gain difference.
* **`response_tau` / `actuator_delay`** — the rack's first order time constant and dead
  time. These trade off against each other under smooth steering, so prefer their sum,
  `effective_lag`, wherever you only care about how late the car is.
* **`deadzone` / `max_useful_torque`** — where the EPS actually starts and stops
  responding, i.e. how much of `STEER_MAX` is real.
* **`steer_ratio` / `understeer_gradient`** — when an independent yaw source is available.
* **`driver_torque_threshold`** — a learned replacement for `STEER_THRESHOLD`.

## How

The steady state model is the one openpilot's torque control already uses, made per-Honda
and speed scheduled:

```
u = a_lat / K(v) + friction * sign(rack rate) + offset + asymmetry * max(a_lat, 0)
```

fitted by recursive least squares with a forgetting factor, seeded from a per-platform
prior derived from the car's own `CarParams` (`prior_from_car_params`), so an unlearned
car behaves exactly as it does today.

Three details make it work on the gentle steering openpilot actually does:

1. **Delay is identified by model fit, not correlation.** A bank of copies of the model is
   fitted at candidate dead times, and the candidate that explains the command best wins;
   its fit is the one that is reported. Cross correlating command against steering rate is
   cheaper but needs excitation that lane keeping does not provide.
2. **Static and lag terms are fitted on different samples.** `sign(ȧ)` and `ȧ` are strongly
   correlated over smooth steering, so fitting friction and lag together splits the truth
   arbitrarily between them. Friction, gain and bias are fitted only where the car is
   settled (`|ȧ| ≤ 0.5 m/s³`); the lag term is fitted only where it is clearly moving, on
   the residual left after the static model.
3. **Regressors are filtered before they are trusted.** Lateral acceleration is
   differentiated over 0.1 s rather than sample to sample, and the friction sign comes from
   a filtered rack speed. Noise in a regressor biases its own coefficient toward zero, and
   at 100 Hz a raw difference is nearly all noise.

Samples are bucketed by speed × lateral acceleration and the model only reports `valid`
once several cells are populated, so a long highway straight cannot on its own convince
the learner it knows the car.

## Using it

`lat_controller.py` is a PI + learned feedforward lateral controller built on the model.
Feedback gains are specified in plant independent units and divided by the learned gain, so
one set of numbers behaves the same on an N-Box and a Pilot, and the learned model is
blended in over 60 s of validity, so an unlearned car drives like its prior.

```python
from opendbc.car.honda.lat_controller import HondaAdaptiveLatController

ctrl = HondaAdaptiveLatController(CP, learned=stored_model)     # stored_model may be None
torque, dbg = ctrl.update(desired_lat_accel, measured_lat_accel, v_ego,
                          CS.steeringAngleDeg, CS.steeringRateDeg,
                          lat_active=CC.latActive, steering_pressed=CS.steeringPressed,
                          driver_torque=CS.steeringTorque, yaw_rate=yaw_rate,
                          desired_lat_jerk=desired_lat_jerk)
apply_torque = int(torque * params.STEER_MAX)
stored_model = ctrl.learner.model()                             # persist between drives
```

The learner can also be driven on its own (`HondaSteeringLearner.update(HondaSteerSample(...))`)
next to the existing controller, to collect models without changing what the car does.

This is deliberately *not* wired into the shipping Honda control path: it is a controller
and an identification tool, and swapping the lateral controller for the whole Honda fleet
is a change that should be made on measured routes, not on a merge.

## Offline, over the fleet

`tools/car_porting/learn_honda_steering.py` in openpilot replays routes through the same
learner:

```
./learn_honda_steering.py --all --table            # every Honda route in routes.py
./learn_honda_steering.py <route> -v               # one route, with sample counts
./learn_honda_steering.py --all --out models.json  # dump the learned models
```

It prints each learned quantity next to the value the platform is currently hardcoded with,
which is the fastest way to see which of the tables above are wrong and by how much.

## Tests

`tests/test_steering_learner.py` builds a plant per platform from that platform's own
`CarSpecs` (`tests/honda_steer_plant.py`), with truth deliberately displaced from the prior,
and checks that the learner recovers it — for *every* car in `values.py`, plus that learning
improves tracking, that nothing is learned while disengaged or overridden, and that the
command stays bounded whatever the model says.
