from typing import Iterable, Iterator, TypeVar, Callable, Sequence, Tuple, Dict
import numpy as np

S = TypeVar('S')
A = TypeVar('A')


def lspi(
        data: Iterable[Tuple[S, A, float, S]],
        features: Sequence[Callable[[S, A], float]],
        actions: Dict[S, Sequence[A]],
        weights_0: np.ndarray,
        γ: float,
        tolerance: float
) -> Iterator[Callable[[S], A]]:
    '''
    Learns a policy from data
    Finds the weights 'w' involved in the linear
    function approximation of Q:(SxA)->float.
    Returns the policy function for a state 's' given by
    the argmax over actions all actions 'a' of Q(s,a;w).
    '''

    def policy(s: S, w: np.ndarray) -> A:
        q = []
        if s not in actions:
            return None
        for a in actions[s]:
            f = [phi(s, a) for phi in features]
            q.append(np.dot(f, w))
        a_star_index: int = np.argmax(q)[0]
        return actions[s][a_star_index]

    def lstdq(w: np.ndarray) -> np.ndarray:
        A: np.ndarray = np.zeros((m, m))
        b: np.ndarray = np.zeros(m)

        for step in data:
            s: S = step[0]
            a: A = step[1]
            r: float = step[2]
            s1: S = step[3]
            policy_s1: np.ndarray = policy(s1, w)
            features_s: np.ndarray = np.array([f(s, a) for f in features])
            if policy_s1 is not None:
                features_s1: np.ndarray = np.array([f(s1, policy_s1) for f in features])
            else:
                features_s1: np.ndarray = np.zeros(m)
            A += np.outer(features_s, features_s - γ * features_s1)
            b += r * features_s

        weights: np.ndarray = np.dot(np.linalg.pinv(A), b)
        return weights

    m: int = len(features)
    epsilon: float = tolerance * 1e6
    weights = weights_0
    while epsilon >= tolerance:
        old_weights = weights
        weights = lstdq(old_weights)
        epsilon = max(abs(old_weights[i]-weights[i]) for i in range(m))
        yield lambda s: policy(s, weights)
