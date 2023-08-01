from typing import Any, Callable, Optional, Tuple, Type, Sequence, NamedTuple, Set
import uuid
from datetime import datetime, timedelta
from copy import deepcopy
from scipy.stats import multivariate_normal

import itertools

import numpy as np
import math
import opt
import time

from models import CombinedTransitionModel, ConstantVelocity, MeasurementModel


class Elipse(NamedTuple):
    center: Sequence[float]
    width: float
    height: float
    angle: float


class Gaussian(NamedTuple):
    mu: np.ndarray
    covar: np.ndarray

    def Elipse(self, mapping=None):
        if mapping is None:
            w, v = np.linalg.eig(self.covar)
        else:
            HH = np.eye(self.mu.size)[(0, 2), :]
            w, v = np.linalg.eig(HH @ self.covar @ HH.T)

        max_ind = np.argmax(w)
        min_ind = np.argmin(w)
        orient = np.arctan2(v[1, max_ind], v[0, max_ind])

        if mapping is None:
            center = self.mu.reshape(-1).tolist()
        else:
            center = self.mu[mapping, 0].tolist()

        width = float(2 * np.sqrt(w[max_ind]))
        height = float(2 * np.sqrt(w[min_ind]))
        angle = float(np.rad2deg(orient))

        return Elipse(center, width, height, angle)


class HypEntry(NamedTuple):
    prob: float
    gauss: Gaussian
    name: str
    start_time: int
    hist: Sequence[Tuple[int, int]]


class Hyp(NamedTuple):
    weight: float
    entries: Sequence[HypEntry]


class PPP(NamedTuple):
    weight: float
    gauss: Gaussian


class State(NamedTuple):
    hyps: Sequence[Hyp] = []


def softmax(xs):
    xs -= xs.max()
    xs = np.exp(xs)
    return xs / xs.sum()


class MBMPredictor(object):
    def __init__(
        self,
        survival_probability,
        transition_model,
        ppps,
    ):
        self._survival_probability = survival_probability
        self._transition_model = transition_model
        self._ppps = ppps

    def __call__(self, state, current_time, time_interval):
        F = self._transition_model.F(time_interval=time_interval)
        Q = self._transition_model.Q(time_interval=time_interval)

        new_entries = [
            HypEntry(
                prob=weight,
                gauss=Gaussian(
                    mu=mu,
                    covar=covar,
                ),
                name=str(uuid.uuid4()),
                start_time=current_time,
                hist=[],
            ) for weight, (mu, covar) in self._ppps
        ]

        if state.hyps:
            hyps = [
                Hyp(
                    weight=weight,
                    entries=[
                        HypEntry(
                            prob=prob * self._survival_probability,
                            gauss=Gaussian(
                                mu=F @ mu,
                                covar=F @ covar @ F.T + Q,
                            ),
                            name=name,
                            start_time=start_time,
                            hist=hist,
                        ) for prob, (mu,
                                     covar), name, start_time, hist in entries
                    ] + deepcopy(new_entries),
                ) for weight, entries in state.hyps
            ]
        else:
            hyps = [
                Hyp(
                    weight=1.0,
                    entries=new_entries,
                ),
            ]

        return State(hyps)


class MBMUpdater(object):
    def __init__(
        self,
        detection_probability,
        clutter_intensity,
        gating_threshold,
        num_max_hyps,
        H,
        R,
    ):
        self._detection_probability = detection_probability
        self._misdetection_probability = 1. - self._detection_probability
        self._gating_threshold = gating_threshold
        self._clutter_intensity = clutter_intensity
        self._num_max_hyps = num_max_hyps

        self._H = H
        self._R = R

    def __call__(self, state, zs, current_time):
        hyps = list(
            itertools.chain(*[
                self._hyp_to_hyps(hyp, zs, current_time) for hyp in state.hyps
            ]))
        weights = np.array(
            [weight for weight, _ in hyps],
            dtype=np.float32,
        )
        weights = softmax(weights).tolist()
        hyps = [
            Hyp(
                weight=weights[ind],
                entries=hyps[ind][1],
            ) for ind in np.argsort(weights)[::-1].tolist()
        ]

        return State(hyps)

    def _hyp_to_hyps(self, hyp, zs, current_time):
        num_tracks = len(hyp.entries)
        num_measurements = len(zs)

        cost = np.full(
            shape=(num_tracks, num_measurements + num_tracks),
            fill_value=np.finfo('float32').min,
            dtype=np.float32,
        )

        entry_to_entries = [None] * num_tracks
        for i, entry in enumerate(hyp.entries):
            misdetection_entry, detection_entries = self._hyp_entry_to_hyp_entries(
                entry, zs, current_time)
            entry_to_entries[i] = (misdetection_entry, detection_entries)

            cost[i, num_measurements + i] = misdetection_entry[0]
            for j, detection_entry in enumerate(detection_entries):
                if detection_entry[1] is not None:
                    cost[i, j] = detection_entry[0] - math.log(
                        self._clutter_intensity)

        hyps = []
        for assoc, assoc_cost in zip(
                *opt.murty(cost.tolist(), self._num_max_hyps)):
            for i, j in enumerate(assoc):
                new_entries = []
                if j < num_measurements:
                    new_entries.append(entry_to_entries[i][1][j][1])
                else:
                    new_entries.append(entry_to_entries[i][0][1])

                weight = math.log(
                    hyp.weight) + assoc_cost + num_measurements * math.log(
                        self._clutter_intensity)
                hyps.append(Hyp(
                    weight=weight,
                    entries=new_entries,
                ))
        return hyps

    def _hyp_entry_to_hyp_entries(self, hyp, zs, current_time):
        Nz, Nx = np.shape(self._H)

        prob, (mu, covar), name, start_time, hist = hyp

        misdetection_hyp = (
            math.log(1. - prob + prob * self._misdetection_probability),
            HypEntry(
                prob=prob * self._misdetection_probability /
                (1. - prob + prob * self._misdetection_probability),
                gauss=Gaussian(
                    mu=mu,
                    covar=covar,
                ),
                name=name,
                start_time=start_time,
                hist=hist + [(-1, current_time)],
            ),
        )

        S = self._H @ covar @ self._H.T  # NzXNz
        z_pred = self._H @ mu  #Nzx1

        V = np.linalg.cholesky(S)  # NzXNz
        S_inv_sqrt = np.linalg.inv(V)  # NzXNz
        S_inv = S_inv_sqrt.T @ S_inv_sqrt  # NzXNz
        S_log_det = 2 * np.log(np.prod(np.diag(V)))  # float

        K = covar @ self._H.T @ S_inv  #NxXNz

        hyps = []
        for i, z in enumerate(zs):
            z_minus_z_pred = z - z_pred  # NzX1
            distance = float(z_minus_z_pred.T @ S_inv @ z_minus_z_pred)
            if distance < self._gating_threshold:
                P = (np.eye(Nx) - K @ self._H) @ covar  #NxXNx
                P = (P + P.T) / 2  # NxXNx
                x = mu + K @ z_minus_z_pred  #NxX1

                hyps.append((
                    math.log(prob * self._detection_probability) - 0.5 *
                    (distance + S_log_det + Nz * math.log(math.pi * 2)),
                    HypEntry(
                        prob=1.,
                        gauss=Gaussian(
                            mu=x,
                            covar=P,
                        ),
                        name=name,
                        start_time=start_time,
                        hist=hist + [(i, current_time)],
                    ),
                ))
            else:
                hyps.append((float('-inf'), None))
        return misdetection_hyp, hyps


class MBMPruner(object):
    def __init__(
        self,
        hyp_pruning_threshold,
        existence_pruning_threshold,
        num_max_hyps,
    ):
        self._hyp_pruning_threshold = hyp_pruning_threshold
        self._existence_pruning_threshold = existence_pruning_threshold
        self._num_max_hyps = num_max_hyps

    def __call__(self, state):
        hyps = [
            Hyp(
                weight=weight,
                entries=[
                    entry for entry in entries
                    if entry.prob >= self._existence_pruning_threshold
                ],
            ) for weight, entries in state.hyps
            if weight >= self._hyp_pruning_threshold
        ]

        hyps = [hyp for hyp in hyps
                if len(hyp.entries) > 0][:self._num_max_hyps]

        normalizer = sum([
            weight
            for weight, _ in hyps
        ])

        hyps = [
            Hyp(
                weight=weight / normalizer,
                entries=entries,
            )
            for weight, entries in hyps
        ]

        return State(hyps)
