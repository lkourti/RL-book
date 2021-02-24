from typing import Iterable, Iterator, TypeVar, Dict
from rl.function_approx import FunctionApprox
from rl.returns import returns
import rl.markov_process as mp
import rl.iterate as iterate
import itertools

S = TypeVar('S')

def evaluate_mrp_tabular_tdl(
        transitions: Iterable[mp.TransitionStep[S]],
        vf: Dict[S, float],
        γ: float,
        λ: float
) -> Iterator[Dict[S, float]]:

    initial_learning_rate: float = 0.03
    half_life: float = 1000.0
    exponent: float = 0.5
    occurrence: Dict[S, int] = {}

    eligibility_traces: Dict[S, float] = {s: 0 for s in vf}

    for step in transitions:
        state = step.state
        reward = step.reward
        state1 = step.next_state
        eligibility_traces.update({s: γ*λ*e for s,e in eligibility_traces.items()})
        eligibility_traces[state] += 1
        if state in occurrence:
            occurrence[state] += 1
        else:
            occurrence[state] = 1
        lr = initial_learning_rate / (1 + ((occurrence[state]-1)/half_life)**exponent)
        vf[state] += lr * (reward + γ*vf[state1] - vf[state]) * eligibility_traces[state]
        yield vf