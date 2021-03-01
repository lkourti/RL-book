from numpy.polynomial.laguerre import lagval
from typing import Callable, Sequence
import numpy as np


def get_price_for_lspi(
    expiry: float,
    payoff: Callable[[float, float], float],
    gamma: float,
    paths: np.ndarray,
    weights: np.ndarray,
    num_dt: int,
    feature_funcs: Sequence[Callable[[int, float], float]]
) -> float:
    num_paths = paths.shape[0]
    prices = np.zeros(num_paths)
    dt = expiry / num_dt
    for path_num, path in enumerate(paths):
        step = 0
        while step <= num_dt:
            t = dt * step
            price_seq = path[:(step + 1)]
            exercise_price = payoff(t, price_seq[-1])
            if step == num_dt:
                continue_price = 0.
            else:
                continue_price = weights.dot([f(step, price_seq) for f in feature_funcs])
            step += 1
            if exercise_price > continue_price:
                prices[path_num] = gamma**(-t) * exercise_price
                step = num_dt + 1
    return np.average(prices)


def lspi_for_am(
    expiry: float,
    num_dt: int,
    num_paths: int,
    num_iters: int,
    payoff: Callable[[float, np.ndarray], float],
    feature_funcs: Sequence[Callable[[int, np.ndarray], float]],
    paths: np.ndarray,
    epsilon: float,
    gamma: float,
) -> float:
    num_features = len(feature_funcs)
    weights = np.zeros(num_features)
    iter_steps = num_paths * num_dt
    dt = expiry / num_dt

    for _ in range(num_iters):
        A = np.zeros((num_features, num_features))
        b = np.zeros(num_features)

        for path_num, path in enumerate(paths):

            for step in range(num_dt):
                t = step * dt
                phi_s = np.array([f(step, path[:(step + 1)]) for f in feature_funcs])
                next_path = path[:(step + 2)]
                phi_sp = np.zeros(num_features)
                reward = 0.
                next_payoff = payoff(t + dt, next_path[-1])

                if step == num_dt - 1:
                    reward = next_payoff
                else:
                    next_phi = np.array([f(step + 1, next_path[-1]) for f in feature_funcs])
                    if next_payoff > weights.dot(next_phi):
                        reward = next_payoff
                    else:
                        phi_sp = next_phi

                A += np.outer(phi_s, phi_s - phi_sp * gamma)
                b += reward * gamma * phi_s

        A /= iter_steps
        A += epsilon * np.eye(num_features)
        b /= iter_steps
        weights = np.linalg.inv(A).dot(b)

    return get_price_for_lspi(
        expiry,
        payoff,
        gamma,
        weights,
        num_dt,
        feature_funcs
    )


if __name__ == '__main__':
    spot_price = 100.0
    strike = 100.0
    is_call: bool = False
    if is_call:
        payoff = lambda _, x: max(x - strike, 0)
    else:
        payoff = lambda _, x: max(strike - x, 0)
    expiry_val = 1.0
    num_dt_val = 100
    num_paths_val = 10000
    num_iters_val = 15
    num_laguerre_val = 3
    epsilon = 1e-3
    gamma = 0.95

    eye = np.eye(num_laguerre_val)

    def laguerre_feature_func(x: float, i: int) -> float:
        xp = x / strike
        return np.exp(-xp / 2) * lagval(xp, eye[i])

    def feature_func(ind: int, x: float, i: int) -> float:
        dt = expiry_val / num_dt_val
        t = ind * dt
        if i == 0:
            fun = 1.
        elif i < num_laguerre_val + 1:
            fun = laguerre_feature_func(x, i - 1)
        elif i == num_laguerre_val + 1:
            fun = np.sin(-t * np.pi / (2. * expiry_val) + np.pi / 2.)
        elif i == num_laguerre_val + 2:
            fun = np.log(expiry_val - t)
        else:
            fun = (t / expiry_val)*(t / expiry_val)
        return fun


    feature_funcs = [lambda t, x, i=i: feature_func(t, x, i) for i in
                     range(num_laguerre_val + 4)]

    def paths() -> np.ndarray:
        # TO DO - how to model price movement
        pass

    paths: np.ndarray = paths()

    am_lspi_price = lspi_for_am(
        expiry_val,
        num_dt_val,
        num_paths_val,
        num_iters_val,
        payoff,
        feature_funcs,
        paths,
        epsilon,
        gamma
    )
