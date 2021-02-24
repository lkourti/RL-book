from typing import Iterable, Iterator, TypeVar, Dict
from rl.function_approx import FunctionApprox
from rl.returns import returns
import rl.markov_process as mp
import rl.iterate as iterate
import itertools

S = TypeVar('S')


def evaluate_mrp_funapprox_bootstrap(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: FunctionApprox[S],
        γ: float,
        n: int
) -> Iterator[FunctionApprox[S]]:
    '''
    n-Step Bootstrapping Prediction
    for the Function Approximation case
    '''
    tolerance: float = γ ** n  # in order to include n rewards in each bootstrap return
    bootstrap_return_steps: Iterator[mp.ReturnStep] = returns(transitions, γ, tolerance)
    bootstr_return_steps_indexed = zip(itertools.count(), bootstrap_return_steps)


    def step(v, indexed_return_step):
        index = indexed_return_step[0]
        step = indexed_return_step[1]
        step_n = next(itertools.islice(bootstrap_return_steps, index + n, None), None)
        return v.update([(step.state,
                          step.return_ + γ**n * v(step_n.next_state))])

    return iterate.accumulate(bootstr_return_steps_indexed, step, initial=approx_0)


def evaluate_mrp_tabular_bootstrap(
        transitions: Iterable[mp.TransitionStep[S]],
        vf: Dict[S, float],
        γ: float,
        n: int
) -> Iterator[Dict[S, float]]:
    '''
    n-Step Bootstrapping Prediction
    for the Tabular case
    '''
    initial_learning_rate: float = 0.03
    half_life: float = 1000.0
    exponent: float = 0.5
    occurrence: Dict[S, int] = {}
    tolerance: float = γ**n      # in order to include n rewards in each bootstrap return
    bootstrap_return_steps: Iterator[mp.ReturnStep] = returns(transitions, γ, tolerance)

    for i, step in enumerate(bootstrap_return_steps):
        state = step.state
        bootstr_return = step.return_
        step_n = next(itertools.islice(bootstrap_return_steps, i+n, None), None)
        state_n = step_n.state
        if state in occurrence:
            occurrence[state] += 1
        else:
            occurrence[state] = 1
        lr = initial_learning_rate / (1 + ((occurrence[state]-1)/half_life)**exponent)
        vf[state] += lr * (bootstr_return + γ**n * vf[state_n] - vf[state])
        yield vf