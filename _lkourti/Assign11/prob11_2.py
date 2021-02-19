from typing import Iterable, Iterator, TypeVar, Dict
import rl.markov_process as mp

S = TypeVar('S')


def evaluate_mrp_dt(
        transitions: Iterable[mp.TransitionStep[S]],
        vf: Dict[S, float],
        γ: float,
) -> Iterator[Dict[S, float]]:

    initial_learning_rate: float = 0.03
    half_life: float = 1000.0
    exponent: float = 0.5
    occurrence: Dict[S, int] = {}
    for step in transitions:
        state = step.state
        reward = step.reward
        state1 = step.next_state
        if state in occurrence:
            occurrence[state] += 1
        else:
            occurrence[state] = 1
        lr = initial_learning_rate / (1 + ((occurrence[state]-1)/half_life)**exponent)
        vf[state] += lr * (reward + γ*vf[state1] - vf[state])
        yield vf