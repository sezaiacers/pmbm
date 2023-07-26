from typing import Any, Callable, Optional, Tuple, Type, Sequence, NamedTuple, Set
import uuid
from datetime import datetime, timedelta
from copy import deepcopy
from scipy.stats import multivariate_normal

import itertools

import numpy as np
import math
import opt

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
    hist: Sequence[int]


class Hyp(NamedTuple):
    weight: float
    entries: Sequence[HypEntry]


class PPP(NamedTuple):
    weight: float
    gauss: Gaussian


class State(NamedTuple):
    hyps: Sequence[Hyp] = []
    ppps: Sequence[PPP] = []


def softmax(xs):
    xs -= xs.max()
    xs = np.exp(xs)
    return xs / xs.sum()


class PMBMPredictor(object):
    def __init__(
        self,
        survival_probability,
        transition_model,
        ppps,
    ):
        self._survival_probability = survival_probability

        self._transition_model = transition_model
        self._ppps = ppps

    def __call__(self, state, time_interval):
        F = self._transition_model.F(time_interval=time_interval)
        Q = self._transition_model.Q(time_interval=time_interval)

        # Predict PPP's
        ppps = [
            PPP(
                weight=weight * self._survival_probability,
                gauss=Gaussian(
                    mu=F @ mu,
                    covar=F @ covar @ F.T + Q,
                ),
            ) for weight, (mu, covar) in state.ppps
        ]
        ppps += deepcopy(self._ppps)

        #  Predict Hyps
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
                    ) for prob, (mu, covar), name, start_time, hist in entries
                ],
            ) for weight, entries in state.hyps
        ]

        return State(hyps, ppps)


class PMBMUpdater(object):
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
        ppps = self._update_ppps(state.ppps)

        ppp_hyp_entries = self._zs_to_hyp_entries(state.ppps, zs, current_time)
        if state.hyps:
            hyps = list(
                itertools.chain(*[
                    self._hyp_to_hyps(hyp, ppp_hyp_entries, zs)
                    for hyp in state.hyps
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
        else:
            hyps = [
                Hyp(
                    weight=1.,
                    entries=[entry for _, entry in ppp_hyp_entries],
                )
            ]
        return State(hyps=hyps, ppps=ppps)

    def _hyp_to_hyps(self, hyp, ppp_hyp_entries, zs):
        num_prev_tracks = len(hyp.entries)
        num_new_tracks = len(ppp_hyp_entries)

        cost = np.full(
            shape=(num_prev_tracks + num_new_tracks, num_new_tracks),
            fill_value=np.finfo('float32').min,
            dtype=np.float32,
        )
        misdetection_cost = np.zeros(num_prev_tracks, dtype=np.float32)

        entry_to_entries = [None] * num_prev_tracks
        hyp_entry = [None] * (num_prev_tracks + num_new_tracks)
        for i, entry in enumerate(hyp.entries):
            misdetection_entry, detection_entries = self._hyp_entry_to_hyp_entries(
                entry, zs)
            entry_to_entries[i] = (misdetection_entry, detection_entries)

            for j, detection_entry in enumerate(detection_entries):
                if detection_entry[1] is not None:
                    cost[i, j] = detection_entry[0] - misdetection_entry[0]
            misdetection_cost[i] = misdetection_entry[0]

            hyp_entry[i] = misdetection_entry[1]

        for i, entry in enumerate(ppp_hyp_entries):
            cost[i + num_prev_tracks, i] = entry[0]
            hyp_entry[i + num_prev_tracks] = entry[1]

        misdetection_cost_sum = float(np.sum(misdetection_cost))

        hyps = []
        for assoc, assoc_cost in zip(
                *opt.murty(cost.T.tolist(), self._num_max_hyps)):
            new_hyp_entry = deepcopy(hyp_entry)
            for j, i in enumerate(assoc):
                if i < num_prev_tracks:
                    new_hyp_entry[i] = entry_to_entries[i][1][j][1]
            weight = assoc_cost + misdetection_cost_sum + math.log(hyp.weight)

            hyps.append(Hyp(
                weight=weight,
                entries=new_hyp_entry,
            ))
        return hyps

    def _hyp_entry_to_hyp_entries(self, hyp, zs):
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
                hist=hist + [-1],
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
                        hist=hist + [i],
                    ),
                ))
            else:
                hyps.append((float('-inf'), None))
        return misdetection_hyp, hyps

    def _zs_to_hyp_entries(self, ppps, zs, current_time):
        return [
            self._z_to_hyp_entry(i, z, ppps, current_time)
            for i, z in enumerate(zs)
        ]

    def _z_to_hyp_entry(self, i, z, ppps, current_time):
        Nz, Nx = np.shape(self._H)
        states = []
        for weight, (mu, covar) in ppps:
            S = self._H @ covar @ self._H.T + self._R  # NzXNz
            S_inv = np.linalg.inv(S)  # NzXNz

            z_pred = self._H @ mu  # NzX1
            z_minus_z_pred = z - z_pred  # NzX1

            distance = float(z_minus_z_pred.T @ S_inv @ z_minus_z_pred)  # real
            if distance < self._gating_threshold:
                K = covar @ self._H.T @ S_inv  #NxXNz

                P = (np.eye(Nx) - K @ self._H) @ covar  #NxXNx
                P = (P + P.T) / 2  # NxXNx
                x = mu + K @ z_minus_z_pred  #NxX1
                w = multivariate_normal.pdf(
                    z.reshape(-1),
                    z_pred.reshape(-1),
                    S,
                ) * (self._detection_probability * weight)

                states.append((w, x, P))

        if states:
            mu = np.zeros((Nx, 1), dtype=np.float32)
            covar = np.zeros((Nx, Nx), dtype=np.float32)
            weight = 0.0

            for w, x, P in states:
                weight = weight + w
                mu = mu + w * x
                covar = covar + w * P + w * (x @ x.T)

            mu = mu / weight
            covar = covar / weight - mu @ mu.T

            prob = weight
            weight = weight + self._clutter_intensity
            prob = prob / weight

            return (
                math.log(weight),
                HypEntry(
                    prob=prob,
                    gauss=Gaussian(
                        mu=mu,
                        covar=covar,
                    ),
                    name=str(uuid.uuid4()),
                    start_time=current_time,
                    hist=[i],
                ),
            )
        else:
            return (
                math.log(self._clutter_intensity),
                HypEntry(
                    prob=0,
                    gauss=Gaussian(
                        mu=np.zeros((Nx, 1), dtype=np.float32),
                        covar=np.zeros((Nx, Nx), dtype=np.float32),
                    ),
                    name=str(uuid.uuid4()),
                    start_time=current_time,
                    hist=[i],
                ),
            )

    def _update_ppps(self, ppps):
        return [
            PPP(
                weight=weight * self._misdetection_probability,
                gauss=Gaussian(
                    mu=mu,
                    covar=covar,
                ),
            ) for weight, (mu, covar) in ppps
        ]


class PMBMPruner(object):
    def __init__(
        self,
        poisson_pruning_threshold,
        hyp_pruning_threshold,
        existence_pruning_threshold,
        num_max_hyps,
    ):
        self._poisson_pruning_threshold = poisson_pruning_threshold
        self._hyp_pruning_threshold = hyp_pruning_threshold
        self._existence_pruning_threshold = existence_pruning_threshold
        self._num_max_hyps = num_max_hyps

    def __call__(self, state):
        ppps = [
            ppp for ppp in state.ppps
            if ppp.weight >= self._poisson_pruning_threshold
        ]
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

        return State(hyps=hyps, ppps=ppps)


'''
time_interval = timedelta(seconds=1)
transition_model = CombinedTransitionModel(
    ConstantVelocity(11),
    ConstantVelocity(12),
)
measurement_model = MeasurementModel(
    ndim=4,
    mapping=(0, 2),
    covar=np.diag([5, 5]).astype(np.float32),
)
ppps = [
    PPP(
        weight=0.1,
        gauss=Gaussian(
            mu=np.array([[10], [0], [10], [0]], dtype=np.float32),
            covar=np.diag([10, 0, 10, 0]),
        ),
    ),
    PPP(
        weight=0.1,
        gauss=Gaussian(
            mu=np.array([[20], [0], [20], [0]], dtype=np.float32),
            covar=np.diag([10, 0, 10, 0]),
        ),
    ),
    PPP(
        weight=0.1,
        gauss=Gaussian(
            mu=np.array([[30], [0], [30], [0]], dtype=np.float32),
            covar=np.diag([10, 0, 10, 0]),
        ),
    ),
]

zs = [
    np.array([[15], [15]], dtype=np.float32),
    np.array([[20], [20]], dtype=np.float32),
    np.array([[250], [30]], dtype=np.float32),
    np.array([[260], [240]], dtype=np.float32),
]

p = PMBMPredictor(
    0.9,
    transition_model,
    ppps,
)

s = State(hyps=[
    Hyp(1.0,
        entries=[
            HypEntry(
                prob=0.7,
                gauss=Gaussian(
                    mu=np.array([[10], [1], [20], [2]], dtype=np.float32),
                    covar=np.diag([10, 3, 4, 20]),
                ),
                name=str(uuid.uuid4()),
                start_time=0,
                hist=[1, 3],
            )
        ])
], )
#s = State()

s = p(s, time_interval)

u = PMBMUpdater(
    0.9,
    0.001,
    20,
    4,
    measurement_model.H(),
    measurement_model.R(),
)

s = u(s, zs, 0)
'''
'''
from sklearn import datasets
from scipy.stats import multivariate_normal

np.random.seed(0)
ndim = 4

z = np.random.normal(size=(ndim, 1))
x = np.random.normal(size=(ndim, 1))
s = datasets.make_spd_matrix(ndim, random_state=0)


def pdf(x, mean, S):
    Nz, _ = x.shape
    V = np.linalg.cholesky(S)  # NzXNz
    S_inv_sqrt = np.linalg.inv(V)  # NzXNz
    S_inv = S_inv_sqrt.T @ S_inv_sqrt  # NzXNz
    S_log_det = 2 * np.log(np.prod(np.diag(V)))  # float

    x_minus_mean = x - mean
    distance = float(x_minus_mean.T @ S_inv @ x_minus_mean)
    return -0.5 * (distance + S_log_det + Nz * math.log(math.pi * 2))
'''
'''
# z = [1.76405235;0.40015721;0.97873798;2.2408932]
# s = [ 1.00665013, -1.41434036, -0.4953264 , -0.43520063;-1.41434036,  3.81637179,  0.58656485,  0.5053143;-0.4953264 ,  0.58656485,  0.87865608,  0.00746661;-0.43520063,  0.5053143 ,  0.00746661,  0.79932913]
# x = [1.86755799;-0.97727788;0.95008842;-0.15135721]
'''
