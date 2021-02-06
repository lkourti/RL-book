from __future__ import annotations
from typing import Iterator, Mapping, Tuple, TypeVar, Sequence, List, Callable
from operator import itemgetter
from rl.distribution import Distribution, Constant
from rl.function_approx import FunctionApprox
from rl.iterate import iterate, converged
from rl.markov_process import MarkovRewardProcess
from rl.markov_decision_process import Policy,MarkovDecisionProcess
from rl.approximate_dynamic_programming import evaluate_mrp


S = TypeVar('S')
A = TypeVar('A')


class ThisPolicy(Policy[S, A]):
    def __init__(self, mdp: MarkovDecisionProcess[S, A], return_: Callable[[Tuple[S, float]], float]):
        self.mdp: MarkovDecisionProcess[S, A] = mdp
        self.return_: Callable[[Tuple[S, float]], float] = return_

    def act(self, state: S) -> Constant[A]:
        return Constant(max(
            ((self.mdp.step(state, a).expectation(self.return_), a)
             for a in self.mdp.actions(state)),
            key=itemgetter(0)
        )[1])


def policy_iteration(
    mdp: MarkovDecisionProcess[S, A],
    γ: float,
    approx_v_0: FunctionApprox[S],
    non_terminal_states_distribution: Distribution[S],
    num_state_samples: int
) -> Iterator[Tuple[FunctionApprox[S], ThisPolicy[S, A]]]:


    def update(vf_policy: Tuple[FunctionApprox[S], ThisPolicy[S, A]]) \
            -> Tuple[FunctionApprox[S], ThisPolicy[S, A]]:

        nt_states: Sequence[S] = non_terminal_states_distribution\
            .sample_n(num_state_samples)

        vf, pi = vf_policy
        mrp: MarkovRewardProcess[S] = mdp.apply_policy(pi)
        new_vf: FunctionApprox[S] = converged(
            evaluate_mrp(mrp, γ, vf, non_terminal_states_distribution, num_state_samples),
            done=lambda a, b: a.within(b, 1e-4)
        )

        def return_(s_r: Tuple[S, float]) -> float:
            s1, r = s_r
            return r + γ * new_vf.evaluate([s1]).item()

        return (new_vf.update([(s, max(mdp.step(s, a).expectation(return_)
                                       for a in mdp.actions(s))) for s in nt_states]),
                ThisPolicy(mdp, return_))

    def return_(s_r: Tuple[S, float]) -> float:
        s1, r = s_r
        return r + γ * approx_v_0.evaluate([s1]).item()

    return iterate(update, (approx_v_0, ThisPolicy(mdp, return_)))