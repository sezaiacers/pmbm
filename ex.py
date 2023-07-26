import cv2 as cv
from detectors import get_detector
from colors import pickColorPalette
from box_point import ElipseCenterBoxPointExtractor, CenterBoxPointExtractor

from itertools import chain

import collections
import numpy as np
from datetime import datetime, timedelta

from models import CombinedTransitionModel, ConstantVelocity, MeasurementModel
from mpmb import PMBMPredictor, PMBMUpdater, PMBMPruner, PPP, Gaussian, State
from colors import pickColorPalette

transition_model = CombinedTransitionModel(
    ConstantVelocity(120),
    ConstantVelocity(120),
)
measurement_model = MeasurementModel(
    ndim=4,
    mapping=(0, 2),
    covar=np.diag([120, 120]).astype(np.float32),
)

ppps = list(chain(*[
    [
        PPP(
            weight=0.01,
            gauss=Gaussian(
                mu=np.array([[400], [0], [300], [0]]) + 100 * np.array([[i], [0], [j], [0]]),
                covar=np.diag([100, 1, 100, 1]),
            ),
        )
        for j in range(-3, 3)
    ]
    for i in range(-4, 4)
]))


predictor = PMBMPredictor(
    survival_probability=0.9,
    transition_model=transition_model,
    ppps=ppps,
)

updater = PMBMUpdater(
    detection_probability=0.94,
    clutter_intensity=2 / (600 * 800),
    gating_threshold=100,
    num_max_hyps=10,
    H=measurement_model.H(),
    R=measurement_model.R(),
)
pruner = PMBMPruner(
    poisson_pruning_threshold=0.0001,
    hyp_pruning_threshold=0.0001,
    existence_pruning_threshold=0.00001,
    num_max_hyps=6,
)


j = -1
def inc():
    global j
    j += 1
    return j
tracks_to_idx = collections.defaultdict(inc)
#extractor = ElipseCenterBoxPointExtractor()


cap = cv.VideoCapture('/Users/sezaiacers/videos/b.mp4')
detector = get_detector(
    'CenterNetV1',
    '/Users/sezaiacers/models/antigone_v1/model.tflite',
    0.3,
    center_crop=True,
)




start_time = datetime.now()

i = 0
extractor = CenterBoxPointExtractor()
time_interval = timedelta(milliseconds=(1000 // 8))
state = State()
timestamp_to_zs = {}
while True:
    timestamp = start_time + time_interval * i
    ret, frame = cap.read()
    if not ret:
        break
    boxes, scores, labels = detector(frame[..., ::-1])
    h, w, _ = frame.shape
    zs = []
    for ymin, xmin, ymax, xmax in boxes.tolist():
        cx, cy = extractor.extract(ymin, xmin, ymax, xmax)
        cx, cy = cx * w, cy * h
        zs.append(np.array([[cx], [cy]], dtype=np.float32))

    timestamp_to_zs[timestamp] = zs

        #frame = cv.circle(frame, (int(cx), int(cy)), 2, (255, 0, 255), 2)

    state = predictor(state, time_interval)
    state = updater(state, zs, timestamp)
    state = pruner(state)

    print(len(state.ppps))

    for _, entries in state.hyps:
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
    cv.imshow('image', frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'): break

    i += 1
