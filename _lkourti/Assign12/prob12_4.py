from dataclasses import dataclass
from typing import Tuple, Dict, Iterable, Iterator, Callable
from rl.markov_process import MarkovRewardProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_process import RewardTransition
from rl.markov_process import TransitionStep
from scipy.stats import poisson
from rl.distribution import SampledDistribution, Categorical, Choose
from rl.function_approx import FunctionApprox, Tabular, learning_rate_schedule
from rl.monte_carlo import evaluate_mrp
from prob12_2 import evaluate_mrp_tabular_tdl
from rl.iterate import last
from math import sqrt
from itertools import islice
from rl.td import evaluate_mrp
from pprint import pprint
import itertools
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
    user_lambda = 0.9

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

    traces: Iterable[Iterable[TransitionStep[InventoryState]]] = \
        si_mrp.reward_traces(Choose(set(si_mrp.non_terminal_states)))
    episode_length: int = 100
    unit_experiences_accumulated: Iterable[TransitionStep[InventoryState]] = \
        itertools.chain.from_iterable(
            itertools.islice(trace, episode_length) for trace in traces
        )
    num_episodes = 10000#0

    # print("Value Function (Tabular TD(lambda) from scratch)")
    # print("--------------")
    # td_vfs: Iterator[Dict[InventoryState, float]] = evaluate_mrp_tabular_tdl(
    #     transitions=unit_experiences_accumulated,
    #     vf={s: 0 for s in si_mrp.non_terminal_states},
    #     γ=user_gamma,
    #     λ=user_lambda
    # )
    # final_td_vf: Dict[InventoryState, float] = \
    #     last(itertools.islice(td_vfs, episode_length * num_episodes))
    # pprint({s: round(final_td_vf[s], 3) for s in si_mrp.non_terminal_states})

    # TD(λ) convergence
    import matplotlib.pyplot as plt
    colors: list = ['b', 'g', 'r']
    plt.figure(figsize=(11, 7))
    true_vf = si_mrp.get_value_function_vec(user_gamma)
    states = si_mrp.non_terminal_states
    num_states = len(states)
    plot_batch = episode_length
    for index, l in enumerate([0.2, 0.5, 0.8]):
        td_vf_it: Iterator[Dict[InventoryState, float]] = evaluate_mrp_tabular_tdl(
            transitions=unit_experiences_accumulated,
            vf={s: 0 for s in si_mrp.non_terminal_states},
            γ=user_gamma,
            λ=l
        )
        errors = []
        batch_errors = []
        transitions_batch = plot_batch #* episode_length
        for i, td_vf in enumerate(itertools.islice(td_vf_it, episode_length * num_episodes)):
            errors.append(sqrt(sum(
                (td_vf[s] - true_vf[j]) ** 2 for j, s in enumerate(states)
            ) / len(states)))
            # if i % transitions_batch == transitions_batch - 1:
            #     errors.append(sum(batch_errors) / transitions_batch)
            #     batch_td_errs = []
        #plt.figure(figsize=(11, 7))
        label = f"λ={l:.3f}"
        plt.plot(
            np.arange(30000, len(errors)),
            errors[30000:],
            linewidth=0.5,
            color=colors[index],
            linestyle='-',
            label=label
        )

    plt.xlabel("Individual Experiences", fontsize=20)
    plt.ylabel("Value Function RMSE", fontsize=20)
    plt.title(
        "RMSE of TD(lambda) as function of individual experiences",
        fontsize=25
    )
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.show()




