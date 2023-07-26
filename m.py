import datetime
from itertools import tee, chain

import numpy as np
import stonesoup
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState

import tqdm

initial_state_mean = StateVector([[0], [0], [0], [0]])
initial_state_covariance = CovarianceMatrix(np.diag([4, 0.5, 4, 0.5]))
timestep_size = datetime.timedelta(seconds=5)
number_of_steps = 200
birth_rate = 0.3
death_probability = 0.05
initial_state = GaussianState(initial_state_mean, initial_state_covariance)

from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])

from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator

sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    timestep=timestep_size,
    number_steps=number_of_steps,
    birth_rate=birth_rate,
    death_probability=death_probability,
    initial_number_targets=100,
)

from stonesoup.simulator.simple import SimpleDetectionSimulator
from stonesoup.models.measurement.linear import LinearGaussian

# initialise the measurement model
measurement_model_covariance = np.diag([0.25, 0.25])
measurement_model = LinearGaussian(4, [0, 2], measurement_model_covariance)

# probability of detection
probability_detection = 0.9

# clutter will be generated uniformly in this are around the target
clutter_area = np.array([[-1, 1], [-1, 1]]) * 500
clutter_rate = 10
clutter_spatial_density = clutter_rate / np.prod(np.diff(clutter_area))

detection_sims = SimpleDetectionSimulator(
    groundtruth=sim,
    measurement_model=measurement_model,
    detection_probability=probability_detection,
    meas_range=clutter_area,
    clutter_rate=clutter_rate,
)

stonesoup.types.detection.Clutter
stonesoup.types.detection.TrueDetection

zss = []
x = np.ones((2, 1)) * 500
for current_time, measurements in detection_sims:
    zs = []
    for measurement in measurements:
        if isinstance(measurement, stonesoup.types.detection.Clutter):
            zs.append(('C', measurement.state_vector + x))
        elif isinstance(measurement, stonesoup.types.detection.TrueDetection):
            zs.append(('D', measurement.state_vector + x))
        else:
            raise ValueError()
    zss.append((current_time, zs))

from models import CombinedTransitionModel, ConstantVelocity, MeasurementModel
from mpmb import PMBMPredictor, PMBMUpdater, PMBMPruner, PPP, Gaussian, State

transition_model = CombinedTransitionModel(
    ConstantVelocity(0.05),
    ConstantVelocity(0.05),
)
measurement_model = MeasurementModel(
    ndim=4,
    mapping=(0, 2),
    covar=np.diag([0.25, 0.25]).astype(np.float32),
)

ppps = list(
    chain(*[[
        PPP(
            weight=0.001,
            gauss=Gaussian(
                mu=np.array([[500], [0], [500], [0]]) +
                50 * np.array([[i], [0], [j], [0]]),
                covar=np.diag([100, 1, 100, 1]),
            ),
        ) for j in range(-4, 4)
    ] for i in range(-4, 4)]))

predictor = PMBMPredictor(
    1. - death_probability,
    transition_model,
    ppps,
)

updater = PMBMUpdater(
    detection_probability=probability_detection,
    clutter_intensity=clutter_spatial_density,
    gating_threshold=20,
    num_max_hyps=4,
    H=measurement_model.H(),
    R=measurement_model.R(),
)

pruner = PMBMPruner(
    poisson_pruning_threshold=0.0001,
    hyp_pruning_threshold=0.001,
    existence_pruning_threshold=0.001,
    num_max_hyps=2,
)

states = []
s = State()
prev_time = zss[0][0]
timestamp_to_zs = {}
for time, zs in tqdm.tqdm(zss):
    time_interval = time - prev_time
    zs = [np.array(z) for _, z in zs]
    timestamp_to_zs[time] = zs

    s = predictor(s, time_interval)
    s = updater(s, zs, time)
    s = pruner(s)

    states.append(s)

    prev_time = time

import cv2 as cv
from colors import pickColorPalette

i, size = 0, len(zss)
while True:
    frame = np.full(
        shape=(1200, 1200, 3),
        fill_value=255,
        dtype=np.uint8,
    )

    for _, entries in states[i].hyps:
        for entry in entries:
            color = pickColorPalette(hash(entry.name))
            prev_z = None
            for ind, z_time in entry.hist:
                if ind == -1: continue
                z = timestamp_to_zs[z_time][ind].astype(np.int32).reshape(-1).tolist()
                if prev_z is not None:
                    frame = cv.line(frame, prev_z, z, color, 5)
                    pass
                prev_z = z

    cv.imshow('frame', frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        i += 1
        if i == size:
            i = 0
    elif key == ord('p'):
        i -= 1
        if i < 0:
            i = size - 1
