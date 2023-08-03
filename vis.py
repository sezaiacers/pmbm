import cv2 as cv
from detectors import get_detector
from colors import pickColorPalette
from box_point import ElipseCenterBoxPointExtractor, CenterBoxPointExtractor, BottomBoxPointExtractor

from itertools import chain
import time

import math
import collections
import numpy as np
from datetime import datetime, timedelta

from models import CombinedTransitionModel, ConstantVelocity, MeasurementModel
from pmbm import PMBMPredictor, PMBMUpdater, PMBMPruner, PPP, Gaussian, State
from mbm import MBMPredictor, MBMUpdater, MBMPruner
from colors import pickColorPalette

transition_model = CombinedTransitionModel(
    ConstantVelocity(200),
    ConstantVelocity(200),
)

measurement_model = MeasurementModel(
    ndim=4,
    mapping=(0, 2),
    covar=np.diag([200, 200]).astype(np.float32),
)

h, w = 600, 800
nh, nw = 8, 8

ppps = [[
    PPP(
        weight=0.01,
        gauss=Gaussian(
            mu=np.array([[(i + 0.5) * (w / nw)], [0], [(j + 0.5) *
                                                       (h / nh)], [0]],
                        dtype=np.float32),
            covar=np.diag([w/ nw, 1, h / nw, 1]).astype(np.float32),
        ),
    ) for j in range(nh)
] for i in range(nw)]

ppps = list(chain(*ppps))
num_max_hyps = 4
assert round(math.sqrt(num_max_hyps)) * round(
    math.sqrt(num_max_hyps)) == num_max_hyps

predictor = PMBMPredictor(
    survival_probability=0.96,
    transition_model=transition_model,
    ppps=ppps,
)

updater = PMBMUpdater(
    detection_probability=0.95,
    clutter_intensity=2 / (600 * 800),
    gating_threshold=10,
    num_max_hyps=10,
    H=measurement_model.H(),
    R=measurement_model.R(),
)
pruner = PMBMPruner(
    poisson_pruning_threshold=0.00001,
    hyp_pruning_threshold=0.000001,
    existence_pruning_threshold=0.00001,
    num_max_hyps=num_max_hyps,
)

cap = cv.VideoCapture('/Users/sezaiacers/wds/tav/tav/zagreb/a.mp4')
detector = get_detector(
    'CenterNetV1',
    '/Users/sezaiacers/models/antigone_v1/model.tflite',
    0.1,
    center_crop=False,
)

i = 0
extractor = CenterBoxPointExtractor()
time_interval = timedelta(seconds=2)
state = State()

max_index = 1000

zss = []
predicted_states = []
updated_states = []

start_time = datetime.now()
for index in range(max_index):
    cap.set(cv.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    assert ret

    boxes, _, _ = detector(frame[..., ::-1])
    h, w, _ = frame.shape
    zs = []
    for ymin, xmin, ymax, xmax in boxes.tolist():
        cx, cy = extractor.extract(ymin, xmin, ymax, xmax)
        cx, cy = cx * w, cy * h
        zs.append(np.array([[cx], [cy]], dtype=np.float32))

    zss.append(zs)

    state = predictor(state, time_interval)
    predicted_states.append(state)

    state = updater(state, zs, index)
    state = pruner(state)

    updated_states.append(state)

current_idx = 0
draw_z = True
draw_gate = False
draw_preds = False
draw_updates = False
draw_pred_ppp = False
draw_up_pp = False

max_index -= 1  # make sure safe
while True:
    cap.set(cv.CAP_PROP_POS_FRAMES, current_idx)
    ret, frame = cap.read()
    assert ret

    frames = [frame.copy() for i in range(num_max_hyps)]

    if draw_z:
        for z in zss[current_idx]:
            z = z.astype(np.int32).reshape(-1).tolist()
            frames = [
                cv.circle(frame, z, 2, (255, 0, 255), 2) for frame in frames
            ]

    if draw_preds:
        for i, (weight,
                entries) in enumerate(predicted_states[current_idx].hyps):
            for prob, gauss, name, start_time, hist in entries:
                color = pickColorPalette(hash(name))
                (cx, cy), w, h, angle = gauss.Elipse(mapping=(0, 2))

                frames[i] = cv.ellipse(frames[i], (int(cx), int(cy)),
                                       (int(w), int(h)), angle, 0, 360, color,
                                       4)

                frames[i] = cv.putText(
                    frames[i],
                    '{}'.format(int(prob * 100)),
                    (int(cx), int(cy)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

            frames[i] = cv.putText(
                frames[i],
                '{:.4f}'.format(weight),
                (10, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                1.,
                (255, 0, 255),
                4,
            )

    if draw_updates:
        for i, (weight,
                entries) in enumerate(updated_states[current_idx].hyps):
            for prob, gauss, name, start_time, hist in entries:
                color = pickColorPalette(hash(name))
                (cx, cy), w, h, angle = gauss.Elipse(mapping=(0, 2))

                frames[i] = cv.putText(
                    frames[i],
                    '{}'.format(int(prob * 100)),
                    (int(cx), int(cy)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
                '''
                try:
                    frames[i] = cv.ellipse(frames[i], (int(cx), int(cy)),
                                           (int(w), int(h)), angle, 0, 360,
                                           color, 4)
                except:
                    pass
                '''

                prev_z = None
                for ind, z_time in hist:
                    if ind == -1: continue
                    z = zss[z_time][ind].astype(np.int32).reshape(-1).tolist()
                    if prev_z is not None:
                        frames[i] = cv.line(frames[i], prev_z, z, color, 4)
                    prev_z = z

            frames[i] = cv.putText(
                frames[i],
                '{:.4f}'.format(weight),
                (10, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                1.,
                (255, 0, 255),
                4,
            )
    if draw_gate:
        for i, (weight,
                entries) in enumerate(predicted_states[current_idx].hyps):
            for prob, (mu, covar), name, start_time, hist in entries:
                color = pickColorPalette(hash(name))

                (cx, cy), w, h, angle = Gaussian(mu,
                                                 covar).Elipse(mapping=(0, 2))

                S = updater._H @ covar @ updater._H.T
                z_pred = updater._H @ mu

                V = np.linalg.cholesky(S)
                S_inv_sqrt = np.linalg.inv(V)
                S_inv = S_inv_sqrt.T @ S_inv_sqrt
                S_log_det = 2 * np.log(np.prod(np.diag(V)))
                K = covar @ updater._H.T @ S_inv

                for z in zss[current_idx]:
                    z_minus_z_pred = z - z_pred
                    distance = float(z_minus_z_pred.T @ S_inv @ z_minus_z_pred)
                    if distance < updater._gating_threshold:
                        z = z.astype(np.int32).reshape(-1).tolist()

                        frames[i] = cv.line(frames[i], (int(cx), int(cy)), z,
                                            color, 4)

    if draw_pred_ppp:
        for weight, gauss in predicted_states[current_idx].ppps:
            color = (0, 0, 0)
            (cx, cy), w, h, angle = gauss.Elipse(mapping=(0, 2))
            for i in range(len(frames)):
                try:
                    frames[i] = cv.ellipse(frames[i], (int(cx), int(cy)),
                                           (int(w), int(h)), angle, 0, 360,
                                           color, 4)
                except:
                    pass

    if draw_up_pp:
        for weight, gauss in updated_states[current_idx].ppps:
            color = (0, 0, 0)
            (cx, cy), w, h, angle = gauss.Elipse(mapping=(0, 2))
            for i in range(len(frames)):
                try:
                    frames[i] = cv.ellipse(frames[i], (int(cx), int(cy)),
                                           (int(w), int(h)), angle, 0, 360,
                                           color, 4)
                except:
                    pass

    n = len(frames)
    sqrt_n = int(math.sqrt(n))

    frames = [
        np.concatenate(frames[i * sqrt_n:(i + 1) * sqrt_n], 1)
        for i in range(sqrt_n)
    ]

    frame = np.concatenate(frames, 0)

    cv.imshow('image', frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('j'):
        current_idx += 1
        if current_idx == max_index:
            current_idx = 0
    elif key == ord('k'):
        if current_idx != 0:
            current_idx -= 1
    elif key == ord('p'):
        draw_preds = not draw_preds
    elif key == ord('u'):
        draw_updates = not draw_updates
    elif key == ord('z'):
        draw_z = not draw_z
    elif key == ord('g'):
        draw_gate = not draw_gate
    elif key == ord('n'):
        draw_pred_ppp = not draw_pred_ppp
    elif key == ord('m'):
        draw_up_pp = not draw_up_pp
