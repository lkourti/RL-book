'''Tabular Monte Carlo methods for working with Markov Reward Process
and Markov Decision Processes.

'''

from typing import Iterable, Iterator, Mapping, Tuple, TypeVar, Dict

from rl.distribution import Distribution
from rl.function_approx import FunctionApprox
import rl.markov_process as mp
import rl.markov_decision_process as markov_decision_process
from rl.markov_decision_process import (MarkovDecisionProcess)
from rl.returns import returns
from copy import deepcopy

S = TypeVar('S')
A = TypeVar('A')

def evaluate_mrp_mc(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        vf: Dict[S,float],
        γ: float,
        tolerance: float = 1e-6
) -> Iterator[Dict[S,float]]:

    episodes: Iterator[Iterator[mp.ReturnStep]] = \
        (returns(trace, γ, tolerance) for trace in traces)
    occurrence: Dict[S,int] = {}

    for episode in episodes:
        for return_step in episode:
            old_vf = deepcopy(vf)
            state = return_step.state
            if state in occurrence:
                occurrence[state] += 1
            else:
                occurrence[state] = 1
            weight_f: float = 1/occurrence[state]
            vf[state] = (1-weight_f) * old_vf[state] + weight_f * return_step.return_
        yield vf