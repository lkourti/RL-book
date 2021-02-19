from dataclasses import dataclass
from typing import Tuple, Dict, Iterable, Iterator
from rl.markov_process import MarkovRewardProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_process import RewardTransition
from rl.markov_process import TransitionStep
from scipy.stats import poisson
from rl.distribution import SampledDistribution, Categorical, Choose
from rl.function_approx import FunctionApprox, Tabular
from rl.monte_carlo import evaluate_mrp
from prob11_1 import evaluate_mrp_mc
from rl.iterate import last
from itertools import islice
from pprint import pprint
import numpy as np

@dataclass(frozen=True)
class InventoryState:
    on_hand: int
    on_order: int

    def inventory_position(self) -> int:
        return self.on_hand + self.on_order


class SimpleInventoryMRP(MarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

    def transition_reward(
        self,
        state: InventoryState
    ) -> SampledDistribution[Tuple[InventoryState, float]]:

        def sample_next_state_reward(state=state) ->\
                Tuple[InventoryState, float]:
            demand_sample: int = np.random.poisson(self.poisson_lambda)
            ip: int = state.inventory_position()
            next_state: InventoryState = InventoryState(
                max(ip - demand_sample, 0),
                max(self.capacity - ip, 0)
            )
            reward: float = - self.holding_cost * state.on_hand\
                - self.stockout_cost * max(demand_sample - ip, 0)
            return next_state, reward

        return SampledDistribution(sample_next_state_reward)


class SimpleInventoryMRPFinite(FiniteMarkovRewardProcess[InventoryState]):

    def __init__(
        self,
        capacity: int,
        poisson_lambda: float,
        holding_cost: float,
        stockout_cost: float
    ):
        self.capacity: int = capacity
        self.poisson_lambda: float = poisson_lambda
        self.holding_cost: float = holding_cost
        self.stockout_cost: float = stockout_cost

        self.poisson_distr = poisson(poisson_lambda)
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> RewardTransition[InventoryState]:
        d: Dict[InventoryState, Categorical[Tuple[InventoryState, float]]] = {}
        for alpha in range(self.capacity + 1):
            for beta in range(self.capacity + 1 - alpha):
                state = InventoryState(alpha, beta)
                ip = state.inventory_position()
                beta1 = self.capacity - ip
                base_reward = - self.holding_cost * state.on_hand
                sr_probs_map: Dict[Tuple[InventoryState, float], float] =\
                    {(InventoryState(ip - i, beta1), base_reward):
                     self.poisson_distr.pmf(i) for i in range(ip)}
                probability = 1 - self.poisson_distr.cdf(ip - 1)
                reward = base_reward - self.stockout_cost *\
                    (probability * (self.poisson_lambda - ip) +
                     ip * self.poisson_distr.pmf(ip))
                sr_probs_map[(InventoryState(0, beta1), reward)] = probability
                d[state] = Categorical(sr_probs_map)
        return d


if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )


    print("Value Function (Exact)")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()

    print("Value Function (MC Function Approximation)")
    print("--------------")
    traces: Iterable[Iterable[TransitionStep[InventoryState]]] = \
        si_mrp.reward_traces(Choose(set(si_mrp.non_terminal_states)))
    it: Iterator[FunctionApprox[InventoryState]] = evaluate_mrp(
        traces=traces,
        approx_0=Tabular(),
        γ=user_gamma
    )
    num_traces = 10000
    last_vf_mc: FunctionApprox[InventoryState] = last(islice(it, num_traces))
    pprint({s: round(last_vf_mc.evaluate([s])[0], 3) for s in si_mrp.non_terminal_states})
    print()

    print("Value Function (Tabular MC from scratch)")
    print("--------------")
    traces: Iterable[Iterable[TransitionStep[InventoryState]]] = \
        si_mrp.reward_traces(Choose(set(si_mrp.non_terminal_states)))
    it: Iterator[Dict[InventoryState, float]] = evaluate_mrp_mc(
        traces=traces,
        vf={s: 0 for s in si_mrp.non_terminal_states},
        γ=user_gamma
    )
    num_traces = 10000
    last_vf_mc: Dict[InventoryState, float] = last(islice(it, num_traces))
    pprint({s: round(last_vf_mc[s],3) for s in si_mrp.non_terminal_states})
