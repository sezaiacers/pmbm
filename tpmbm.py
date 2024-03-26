import typing
import math

import numpy as np
import models

import uuid


class PoissonComponent(typing.NamedTuple):
    weight: float

    mu: np.ndarray
    covar: np.ndarray

    length: int
    birth: int


class Poisson(typing.NamedTuple):
    components: typing.List[PoissonComponent]


class Bernoulli(typing.NamedTuple):
    prob: float

    mu: np.ndarray
    covar: np.ndarray

    length: int
    birth: int

    start: int

    name: str
    hist: typing.Sequence[typing.Tuple[int, int]]


class MultiBernoulli(typing.NamedTuple):
    prob: float
    bernoullis: typing.List[Bernoulli]


class State(typing.NamedTuple):
    poisson: Poisson
    mixtures: typing.List[MultiBernoulli]


'''
def scan(covar, F, Q, lscan):
    Nx, Nx = np.shape(F)

    prev_covar = covar[:lscan, :lscan]  # [N, N, Nx, Nx]
    pred_covar = F @ covar[-1, -1] @ F.T + Q  # [Nx, Nx]

    cross = prev_covar[-1] @ F  # [N, Nx, Nx]

    x = np.concatenate((prev_covar, np.expand_dims(cross, 0)),
                       0)  # [N+1, N, Nx, Nx]
    y = np.concatenate((cross, np.expand_dims(pred_covar, 0)),
                       0)  # [N+1, Nx, Nx]
    z = np.concatenate((x, np.expand_dims(y, 1)), 1)  # [N+1, N+1, Nx, Nx]
    return z
'''


def scan(covar, F, Q, lscan):
    Nx, Nx = np.shape(F)

    prev_covar = covar[-Nx * lscan:, -Nx * lscan:]
    pred_covar = F @ prev_covar[-Nx:, -Nx:] @ F.T + Q

    N, N = np.shape(prev_covar)
    F_mode = np.zeros((Nx, N), dtype=np.float32)
    F_mode[-Nx:, -Nx:] = F

    cross = F_mode @ prev_covar

    return np.concatenate(
        (np.concatenate(
            (prev_covar, cross.T),
            axis=-1,
        ), np.concatenate(
            (cross, pred_covar),
            axis=-1,
        )),
        axis=0,
    )


class Predictor(object):

    def __init__(
        self,
        poisson: Poisson,
        F: np.ndarray,
        Q: np.ndarray,
        ps: float,
        lscan: int,
    ):

        self._poisson = poisson

        self._F = F
        self._Q = Q

        self._ps = ps

        self._lscan = lscan

    def __call__(self, state: State, time: int):
        Nx, Nx = np.shape(F)
        poisson_components = [
            PoissonComponent(
                weight=weight * self._ps,
                mu=np.concatenate((mu, self._F @ mu[-Nx:]), axis=0),
                covar=scan(
                    covar,
                    self._F,
                    self._Q,
                    min(self._lscan - 1, length),
                ),
                length=length + 1,
                birth=birth,
            ) for weight, mu, covar, length, birth in state.poisson
        ]

        poisson_components.extend([
            PoissonComponent(
                weight=weight,
                mu=mu,
                covar=covar,
                birth=time,
                length=1,
            ) for weight, mu, covar, length, birth in self._poisson
        ])

        mixtures = [
            MultiBernoulli(
                prob=prob,
                bernoullis=[
                    Bernoulli(
                        prob=prob * self._ps,
                        mu=np.concatenate((mu, self._F @ mu[-Nx:]), axis=0),
                        covar=scan(
                            covar,
                            self._F,
                            self._Q,
                            min(self._lscan - 1, length),
                        ),
                        length=length + 1,
                        birth=birth,
                        start=start,
                        name=name,
                        hist=hist,
                    ) for prob, mu, covar, length, birth, start, name, hist in
                    bernoullis
                ],
            ) for prob, bernoullis in state.mixtures
        ]

        return State(poisson=poisson, mixtures=mixtures)


class Updater(object):

    def __init__(self, H, R, lscan, pd, cl, gate, num_assoc):
        self._H = H
        self._R = R

        self._lscan = lscan

        self._pd = pd

        self._cl = cl

        self._gate = gate
        self._num_assoc = num_assoc

    def _z_to_bernoulli(self, poisson: Poisson, z: np.ndarray, z_index: int,
                        time: int):
        Nz, Nx = np.shape(self._H)

        component_idx, total_weight, max_weight = -1, 0.0, 0.0

        for idx, (weight, mu, covar, length,
                  birth) in enumerate(poisson.components):

            mu = mu[-Nx:]  # [Nx, 1]
            covar = covar[-Nx:, -Nx:]  # [Nx, Nx]

            S = self._H @ covar @ self._H.T + self._R  #[Nz, Nz]
            S_inv = np.linalg.inv(S)  # [Nz, Nz]

            z_pred = self._H @ mu  # [Nz, 1]
            z_minus_z_pred = z - z_pred  # [Nz, 1]

            distance = float(z_minus_z_pred.T @ S_inv @ z_minus_z_pred)
            if distance < self._gate:
                S_logdet = np.log(np.prod(
                    np.linalg.cholesky(S).diagonal())) * 2
                w = np.exp(-0.5 *
                           (distance + S_logdet + Nz * 1.8378770664093453))
                w *= self._pd * weight

                total_weight += w
                if w > max_weight:
                    component_idx = idx
                    max_weight = w

        if component_idx == -1:
            return (
                math.log(self._cl),
                Bernoulli(
                    prob=0.0,
                    mu=np.zeros((Nx, 1), dtype=np.float32),
                    covar=np.zeros((Nx, Nx), dtype=np.float32),
                    length=0,
                    birth=0,
                    start=0,
                    name='',
                    hist=[],
                ),
            )

        lscan = min(self._lscan, length)

        p_weight, p_mu, p_covar, p_length, p_birth = poisson.components[
            component_idx]

        # recompute
        mu = p_mu[-Nx:]  # [Nx, 1]
        covar = p_covar[-Nx:, -Nx:]  # [Nx, Nx]

        S = self._H @ covar @ self._H.T + self._R  #[Nz, Nz]
        S_inv = np.linalg.inv(S)  # [Nz, Nz]

        z_pred = self._H @ mu  # [Nz, 1]
        z_minus_z_pred = z - z_pred  # [Nz, 1]

        H_T = np.concatenate(
            (np.zeros((Nz, Nx * (lscan - 1)), dtype=np.float32), H),
            axis=-1,
        )  # [Nz, L * Nx]

        mu = p_mu[-Nx * lscan:]  # [L*Nx, 1]
        covar = p_covar[-Nx * lscan:, -Nx * lscan:]  # [L*Nx, L*Nx]

        K = covar @ H_T.T @ S_inv  # [L*Nx, Nz]
        P = (np.eye(lscan * Nx) - K @ H_T) @ covar  # [L*Nx, L*Nx]
        P = (P + P.T) / 2.0

        x = mu + K @ z_minus_z_pred  # [L*Nx, 1]

        weight = total_weight

        prob = weight
        weight = weight + self._cl
        prob = prob / weight
        return (
            math.log(weight),
            Bernoulli(
                prob=prob,
                mu=x,
                covar=P,
                length=p_length,
                birth=p_birth,
                start=time,
                name=str(uuid.uuid4()),
                hist=[(z_index, time)],
            ),
        )

    def __call__(self, state: State, zs: typing.List[np.ndarray], time: int):
        pass


measurement_model = models.MeasurementModel(
    ndim=4,
    mapping=(0, 2),
    covar=np.diag([2, 2]).astype(np.float32),
)
H = measurement_model.H()
R = measurement_model.R()

transition_model = models.CombinedTransitionModel(
    models.ConstantVelocity(600),
    models.ConstantVelocity(600),
)
F = transition_model.F(0.5)
Q = transition_model.Q(0.5)
'''
z_pred_i: [Nz, 1]
S_pred_i: [Nz, Nz]
H_trajectory: [Nz, Nx*3]
mean_i: [Nx*3, 1]
cov_i: [Nx*3, Nx*3]

K_pred_i: cov_i @ inv(H_trajectory.T) @ S_pred_i [Nx*3, Nz]
[Nx*3, Nx*3] @ [Nx*3, Nz] @ [Nz, Nz]

P_u_i=(eye(length(mean_i))-K_pred_i*H_trajectory)*cov_i;
([Nx*3, Nx*3] - [Nx*3, Nz] @ [Nz, Nx*3]) * [Nx*3, Nx*3] [Nx3, Nx3]
'''
