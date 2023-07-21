from typing import Any, Callable, Optional, Tuple, Type, Sequence, NamedTuple
import math
import numpy as np
import copy

from dataclasses import dataclass
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal
import copy

from datetime import datetime, timedelta
import opt
import collections


class TransitionModel(object):
    def F(self, time_interval):
        raise NotImplementedError

    def Q(self, time_interval):
        raise NotImplementedError


class CombinedTransitionModel(TransitionModel):
    def __init__(self, *models):
        super(CombinedTransitionModel, self).__init__()
        self.models = models

    def F(self, time_interval):
        return block_diag(*[model.F(time_interval) for model in self.models])

    def Q(self, time_interval):
        return block_diag(*[model.Q(time_interval) for model in self.models])


class ConstantNthDerivative(TransitionModel):
    def __init__(self, constant_derivative, noise_diff_coeff):
        super(ConstantNthDerivative, self).__init__()

        self.constant_derivative = constant_derivative
        self.noise_diff_coeff = noise_diff_coeff

    def F(self, time_interval):
        time_interval_sec = time_interval.total_seconds()
        N = self.constant_derivative
        Fmat = np.zeros((N + 1, N + 1))
        dt = time_interval_sec
        for i in range(0, N + 1):
            for j in range(i, N + 1):
                Fmat[i, j] = (dt**(j - i)) / math.factorial(j - i)
        return Fmat

    def Q(self, time_interval):
        time_interval_sec = time_interval.total_seconds()
        dt = time_interval_sec
        N = self.constant_derivative
        if N == 1:
            covar = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        else:
            Fmat = self.F(time_interval)
            Q = np.zeros((N + 1, N + 1))
            Q[N, N] = 1
            igrand = Fmat @ Q @ Fmat.T
            covar = np.zeros((N + 1, N + 1))
            for l in range(0, N + 1):
                for k in range(0, N + 1):
                    covar[l, k] = (igrand[l, k] * dt / (1 + N * 2 - l - k))
        covar *= self.noise_diff_coeff
        return covar


class ConstantVelocity(ConstantNthDerivative):
    def __init__(self, noise_diff_coeff):
        super(ConstantVelocity, self).__init__(1, noise_diff_coeff)


class ConstantAcceleration(ConstantNthDerivative):
    def __init__(self, noise_diff_coeff):
        super(ConstantAcceleration, self).__init__(2, noise_diff_coeff)


class MeasurementModel(object):
    def __init__(self, ndim, mapping, covar):
        self.ndim = ndim
        self.mapping = mapping
        self.covar = covar

    def H(self):
        model_matrix = np.zeros((len(self.mapping), self.ndim))
        for dim, state in enumerate(self.mapping):
            if state is not None:
                model_matrix[dim, state] = 1
        return model_matrix

    def R(self):
        return self.covar


@dataclass
class Elipse(object):
    center: Sequence[float]
    width: float
    height: float
    angle: float


@dataclass
class Gaussian(object):
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


class UnTrack(NamedTuple):
    weight: float
    mu: np.ndarray
    covar: np.ndarray


class Hyp(NamedTuple):
    exist_prob: float
    mu: np.ndarray
    covar: np.ndarray
    weight: float
    assoc_hist: Sequence[int]


class Track(NamedTuple):
    hyps: Sequence[Hyp]
    init_time: int


@dataclass
class State(object):
    tracks: Sequence[Track]
    untracks: Sequence[UnTrack]

    hyps: Sequence[Sequence[int]]
    hyp_weights: Sequence[float]


class PMPMUpdater(object):
    def __init__(self, detection_probability, gating_threshold, H, R):
        self._detect_prob = detection_probability
        self._gating_threshold = gating_threshold

        self._H = H
        self._R = R

    def __call__(self, state, zs, current_time):
        new_state = State()

        Nz, Nx = self._H.shape
        undetect_prob = 1. - self._detect_prob

        num_prev_tracks, num_new_tracks = len(state.tracks), len(zs)

        # Poisson update
        new_state.untracks = [
            UnTrack(
                weight=weight * undetect_prob,
                mu=mu,
                covar=covar,
            ) for weight, mu, covar in state.undetected_states
        ]

        # We update all previous tracks with the measurements
        for i, (hyps, init_time) in enumerate(state.tracks):
            new_hyps = [None] * (len(hyps) * (1 + len(zs)))
            new_state.tracks.append(Track(hyps=new_hyps, init_time=init_time))

            for j, (exist_prob, mu, covar, _, assoc_hist) in enumerate(hyps):
                # Misdetection hypotheses
                new_hyps[j] = Hyp(
                    exist_prob=(
                        exist_prob * undetect_prob /
                        (1. - exist_prob + exist_prob * undetect_prob)),
                    mu=mu,
                    covar=covar,
                    weight=(math.log(1. - exist_prob +
                                     exist_prob * undetect_prob)),
                    assoc_hist=assoc_hist + [0],
                )

                # Detection hypotheses
                S = self._H @ covar @ H.T + R
                z_pred = self._H @ mu

                chol_S = np.linalg.cholesky(S)
                log_det_S = 2 * math.log(np.prod(np.diag(chol_S)))
                inv_sqrt_S = np.linalg.inv(chol_S)
                inv_S = inv_sqrt_S.T @ inv_sqrt_S

                K = covar @ self._H.T @ inv_S

                for k, z in enumerate(zs):
                    z_minus_z_pred = z - z_pred

                    d = float(z_minus_z_pred.T @ inv_S @ z_minus_z_pred)
                    if d >= self._gating_threshold: continue

                    hyp_covar = (np.eyes(Nx) - K @ self._H) @ covar
                    hyp_covar = (hyp_covar + hyp_covar.T) * 0.5

                    hyp_mu = mu + K @ z_minus_z_pred

                    hyp_weight = (
                        math.log(exist_prob * self._detect_prob) + \
                        (-0.5 * d) - 1.0 / log_det_S - \
                        Nz * math.log(2 * math.pi) / 2
                    )

                    new_hyps[j + len(hyps) * k] = Hyp(
                        exist_prob=1.0,
                        mu=hyp_mu,
                        covar=hyp_covar,
                        weight=hyp_weight,
                        assoc_hist=assoc_hist + [k],
                    )

        for k, z in enumerate(zs):
            inds = []
            for i, (weight, mu, covar) in enumerate(state.untracks):
                S = self._H @ covar @ self._H.T + self._R
                S_inv = np.linalg.inv(S)
                z_pred = self._H @ mu

                z_minus_z_pred = z - z_pred
                d = float(z_minus_z_pred.T @ S_inv @ z_minus_z_pred)
                if d < self._gating_threshold:
                    inds.append(i)

            if inds:
                hyp_mu = np.zeros((Nx, 1), dtype=np.float32)
                hyp_covar = np.zeros((Nx, Nx), dtype=jnp.float32)
                hyp_weight = 0.0
                for ind in inds:
                    weight, mu, covar = state.untracks[ind]

                    S = self._H @ covar @ self._H.T + self._R
                    K = covar @ self._H.T @ inv_S
                    z_pred = self._H @ mu

                    P = (np.eyes(Nx) - K @ self._H) @ covar
                    P = (P + P.T) * 0.5
                    x = x + K * (z - z_pred)

                    z_t_weight = multivariate_normal.pdf(
                        z,
                        z_pred,
                        S,
                    ) * self._detect_prob * weight

                    hyp_mu += z_t_weight * x
                    hyp_var += z_t_weight * P + z_t_weight * (x @ x.T)
                    hyp_weight += z_t_weight

                hyp_mu = hyp_mu / hyp_weight
                hyp_var = hyp_var / hyp_covar - (hyp_mu @ hyp_mu.T)

                hyp_exist_prob = hyp_weight
                hyp_weight += self._clutter_intensity
                hyp_exist_prob /= hyp_weight

                track = Track(
                    hyps=[
                        Hyp(
                            exist_prob=hyp_exist_prob,
                            mu=hyp_mu,
                            covar=hyp_covar,
                            weight=math.log(hyp_weight),
                            assoc_hist=[k],
                        ),
                    ],
                    init_time=current_time,
                )
            else:
                # The Bernoulli component has existence probability zero
                track = Track(
                    hyps=[
                        Hyp(
                            exist_prob=0.0,
                            mu=np.zeros((Nx, 1), dtype=np.float32),
                            covar=np.zeros((Nx, Nx), dtype=jnp.float32),
                            weight=math.log(self._clutter_intensity),
                            assoc_hist=[k],
                        ),
                    ],
                    init_time=current_time,
                )

            new_state.tracks.append(track)

        if num_prev_tracks == 0:
            new_state.hyp = np.ones((1, num_new_tracks),
                                    dtype=np.int32).tolist()
            new_state.hyp_weights = [1]
            return new_state

        for h, (hyp,
                hyp_weight) in enumerate(zip(state.hyps, state.hyp_weights)):

            cost_assign = np.full(
                shape=(num_prev_tracks + num_new_tracks, num_new_tracks),
                fill_value=float('inf'),
                dtype=np.float32,
            )
            cost_not_assign = np.zeros(num_new_tracks, dtype=np.float32)

            for t in range(num_prev_tracks):
                hyp_ind = hyp[t]
                if hyp_ind is None:
                    raise ValueError()

                num_prev_hyps = len(state.tracks[t].hyps)

                weight = new_state.tracks[t].hyps[hyp_ind].weight
                for k in range(len(zs)):
                    measurement_hyp = new_state.tracks[t].hyps[hyp_ind +
                                                               num_prev_hyps *
                                                               k]
                    if measurement_hyp is None: continue
                    cost_assign[t, k] = measurement_hyp.weight - weight
                cost_not_assign[t] = weight

            for t in range(num_prev_tracks, num_prev_tracks + num_new_tracks):
                measurement_hyp = new_state.tracks[t].hyps[0]
                cost_assign[
                    t, measurement_hyp.assoc_hist[0]] = measurement_hyp.weight

            cost_assign = np.array(x, dtype=np.float32).reshape(-1, 5)
            cost_assign_not_inf = (~np.isinf(cost_assign)).astype(np.int32)
            indices_stay_column, = np.where(np.sum(cost_assign_not_inf, 0) > 1)
            indices_stay_row, = np.where(
                np.sum(cost_assign_not_inf[:, indices_stay_column], 1) > 0)
            fixed_indices_stay_column, = np.where(
                np.sum(cost_assign_not_inf, 0) == 1)
            fixed_indices_stay_row, fixed_indices_stay_column_2 = np.where(
                cost_assign_not_inf[:, fixed_indices_stay_column])
            fixed_indices_stay_row_2 = np.arange(
                fixed_indices_stay_column_2.max() + 1)
            fixed_indices_stay_row_2[
                fixed_indices_stay_column_2] = fixed_indices_stay_row
            fixed_indices_stay_row = fixed_indices_stay_row_2
            fixed_cost = cost_assign[fixed_indices_stay_row,
                                     fixed_indices_stay_column].sum()
            cost_assgn_trimmed = cost_assign[
                indices_stay_row][:, indices_stay_column]

            default_cost = cost_not_assign.sum() + fixed_cost + log(hyp_weight)

            if cost_assgn_trimmed.size == 0:
                assoc_indices = np.full(
                    shape=(1, num_new_tracks),
                    fill_value=-1,
                    dtype=np.int32,
                )
                assoc_indices[
                    fixed_indices_stay_column] = fixed_indices_stay_row
                new_state.hyp_weights.append(default_cost)
            else:
                trimmed_assoc_indices = 0.0
                assoc_costs = []

                assoc_indices = np.full(
                    shape=(2, 4),
                    fill_value=-1,
                    dtype=np.int32,
                )

                for i, trimmed_assoc_indice in enumerate(
                        trimmed_assoc_indices):
                    assoc_indices[i, indices_stay_column] = indices_stay_row[
                        trimmed_assoc_indice]
                    assoc_indices[
                        i, fixed_indices_stay_column] = fixed_indices_stay_row

                for assoc_cost in assoc_costs:
                    new_state.hyp_weights.append(default_cost - assoc_cost)

            new_hyps = np.full(
                shape=(len(assoc_indices), num_prev_tracks + num_new_tracks),
                fill_value=-1,
                dtype=np.int32,
            )
            for t in range(num_prev_tracks):
                hyp_ind = hyp[t]
                if hyp_ind is None:
                    raise ValueError()
                num_prev_hyps = len(state.tracks[t].hyps)
                for a, assoc_indice in enumerate(assoc_indices):
                    try:
                        ind = assoc_indice.index(t)
                        ind = hyp_ind + num_prev_hyps * ind
                    except:
                        ind = hyp_ind

                    new_hyps[a, t] = ind

            for t in range(num_prev_tracks, num_prev_tracks + num_new_tracks):
                for a, assoc_indice in enumerate(assoc_indices):
                    try:
                        assoc_indice.index(t)
                        ind = -1
                    except:
                        ind = 0
                    new_hyps[a, t] = ind
            new_state.hyps.append(new_hyps)


        new_state.hyps = np.concatenate(new_state.hyps, 0)

        hyp_weights = np.array(new_state.hyp_weights, dtype=np.float32)
        hyp_weight_exp = np.exp(hyp_weights - np.max(hyp_weights))
        hyp_weights = hyp_weight_exp / np.sum(hyp_weight_exp)
        new_state.hyp_weights = hyp_weights

        return new_state


class PMPMPruner(object):
    def __init__(self, poisson_pruning_threshold, hyp_pruning_threshold, max_hyp):
        self._poisson_pruning_threshold = poisson_pruning_threshold
        self._hyp_pruning_threshold = hyp_pruning_threshold
        self._max_hyp = max_hyp

    def __call__(self, state):
        new_state = State()

        new_state.untracks = list(
            filter(
                lambda untrack: untrack.weight < self.
                _poisson_pruning_threshold,
                state.undetected_states,
            ))

        mask = state.hyp_weights > self._hyp_pruning_threshold
        hyp_weights = state.hyp_weights[mask]
        hyps = stats.hyps[mask]

        sort_inds = np.argsort(hyp_weights)[::-1]
        hyp_weights = hyp_weights[sort_inds][:self._max_hyp]
        hyps = hyps[sort_inds][:self._max_hyp]
