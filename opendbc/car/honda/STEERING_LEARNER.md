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
* **`max_useful_torque`** — how much of `STEER_MAX` the rack will actually follow. This
  one is *not* learned: it is carried from the prior. A controller that refuses to
  overdrive the rack never generates the evidence that would show where the rack gives up,
  and a limit guessed from the little evidence it does produce is worse than no limit.
* **`steer_ratio` / `understeer_gradient`** — when an independent yaw source is available.
* **`driver_torque_threshold`** — a learned replacement for `STEER_THRESHOLD`.

## How

The steady state model is the one openpilot's torque control already uses, made per-Honda
and speed scheduled:

```
a_lat = K(v) * u  -  K*offset  -  K*friction * sign(rack rate)  -  K*asym * max(u, 0)
```

fitted by recursive least squares with a forgetting factor, seeded from a per-platform
prior derived from the car's own `CarParams` (`prior_from_car_params`), so an unlearned
car behaves exactly as it does today.

**The orientation matters more than anything else here.** The command is the regressor and
the measured lateral acceleration is the dependent variable, matching `torqued`. Written
the other way round — solving for `u` with `a_lat` among the regressors, which this
learner did until route `729a2e65b1f6201d` — puts the noisiest signal in the system on the
regressor side, where its own noise biases its coefficient toward zero. That coefficient
was `1/K`, so the published gain, its reciprocal, ran away *upward*: on that route it
drifted 3.2 → 4.0 → 4.4, hit the 8.0 rail, and finally crossed into negative `1/K`. In
simulation with a 0.8 m/s² measurement error and a stale yaw rate, the old form published
8.0 against a truth of 2.4 while the current form publishes 2.18.

Three further details make it work on the gentle steering openpilot actually does:

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
once several cells are populated **and they span at least 1.3 m/s²**. Point count is not
coverage: the failing route had 1150 points in one cell and 100 in its neighbor, which
passes any count test and still describes a band too narrow to separate gain from offset.

Three things are refused rather than guessed:

* **Asymmetry** stays switched off until the command has been driven both ways. While the
  command is one-signed, `max(u, 0)` is either identical to `u` or identically zero, and
  the pair is unidentifiable — so it is not fitted at all rather than fitted arbitrarily.
* **A diverged fit** — a gain outside 0.2 to 8.0 — is reported as `diverged` and reset to
  the prior, not silently replaced by it. Substituting the prior quietly, which this did
  originally, makes a broken fit and an unfitted one identical in the published model, and
  that is what kept the divergence invisible for a whole route.
* **A stale measurement** is not fitted. `deviceMotion` ran at 7 Hz on the first segment of
  that route against a nominal 20 Hz, and a held yaw rate is simply a wrong lateral
  acceleration. Samples older than 0.1 s still advance the command history — dropping them
  would break the delay alignment, which counts samples — but nothing is fitted to them.

No direction of the covariance may wind up beyond four times its prior width, so a
direction the data never excites cannot grow until the first sample touching it lands with
enormous gain.

## Resuming

A cached model is a starting point *and* the evidence behind it. Passing one as `learned=`
restores:

* each speed bucket's own gain, at its own speed — seeding every bucket from a single
  point on the schedule would flatten the speed dependence the schedule exists for,
* the covariance of every fit, inflated by 2x — resumed confidence, but not full
  confidence, because tires, alignment, load and temperature all change between
  ignitions and a learner that resumes at full confidence cannot notice,
* the bucket coverage and the lifetime point count, so a model that has converged stays
  converged instead of re-earning coverage it already has,
* the dead time, which the delay bank holds until its own race has scored enough
  candidates to mean anything.

The result is a model that compounds across drives rather than restarting each ignition:
two short trips leave a model built on both. A cache that is missing, misshapen, from
another car, or from another model version is skipped rather than trusted — the resume is
lost, the learner is not.

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

The controller is deliberately *not* wired into the shipping Honda control path: swapping
the lateral controller for the whole Honda fleet is a change to make on measured routes,
not on a merge.

The *learner* is wired in, observe-only. openpilot's `hondasteerd` runs it alongside
whatever is actually steering, publishes `hondaSteeringParameters` at 4 Hz, and caches the
model in `Params` across ignition cycles (keyed by fingerprint and model version, so a
different car or a changed model starts over rather than inheriting someone else's
numbers). It costs about 76 us per update at 50 Hz, under half a percent of a core.
Nothing in the control path reads any of it.

Read it back with openpilot's `tools/car_porting/show_honda_steering.py` — from the
device, from a route's logs, or as a per-publish history of how it converged across a
drive.

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
pays for itself where the prior is materially wrong and does no harm where it is not, that
saturated commands do not drag the gain estimate, that nothing is learned while disengaged
or overridden, and that the command stays bounded whatever the model says.
