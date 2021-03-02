from typing import Sequence, TypeVar, Dict, Iterator, Iterable, Callable, Optional, Tuple
from rl.distribution import Distribution
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap
from scipy.stats import bernoulli
from copy import deepcopy
import random

S = TypeVar('S')
A = TypeVar('A')

def find_qvf_sarsa_control(
        states: Iterable[S],
        non_terminal_states: Sequence[S],
        actions: Dict[S, Sequence[A]],
        q_0: Dict[S, Dict[A, float]],  # for all s non terminal and a in actions[s]
        transitions: Callable[[S, A], Optional[Distribution[Tuple[S, float]]]],
        γ: float,
        tolerance: float = 1e-2 # 1e-6
) -> Iterator[Dict[S, Dict[A, float]]]:

    diff: float = tolerance * 1e6
    k: int = 1
    epsilon: float = 1 / k
    qvf_old = q_0
    count_sa: Dict[Tuple[S, A], int] = {}
    initial_learning_rate: float = 0.03
    half_life: float = 1000.0
    exponent: float = 0.5
    while diff >= tolerance:
        qvf = deepcopy(qvf_old)
        s: S = random.choice(non_terminal_states)
        pick_max: bool = bool(bernoulli.rvs(1 - epsilon))
        if pick_max:
            a: A = max(qvf_old[s], key=lambda a: qvf_old[s][a])
        else:
            a: A = random.choice(actions[s])
        s1, r = transitions(s, a).sample()
        pick_max: bool = bool(bernoulli.rvs(1 - epsilon))
        if pick_max:
            a1: A = max(qvf_old[s1], key=lambda a1: qvf_old[s1][a1])
        else:
            a1: A = random.choice(actions[s1])
        if (s, a) not in count_sa:
            count_sa[(s, a)] = 1
        else:
            count_sa[(s, a)] += 1
        lr = initial_learning_rate / (1 + ((count_sa[(s, a)] - 1) / half_life) ** exponent)
        qvf[s][a] += lr * (r + γ * qvf_old[s1][a1] - qvf_old[s][a])
        k += 1
        epsilon = max(abs(qvf[s][a] - qvf_old[s][a]) for s in qvf for a in qvf[s])
        qvf_old = deepcopy(qvf)
        yield qvf

if __name__ == '__main__':
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0
    user_gamma = 0.9

    si_mdp = SimpleInventoryMDPCap(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )

    transition_map = si_mdp.get_action_transition_reward_map()

    def transition_function(s, a):
        return transition_map[s][a]

    states = si_mdp.states()
    non_terminal = si_mdp.non_terminal_states
    actions = {s: list(si_mdp.actions(s)) for s in non_terminal}
    q_0 = {s: {a: 0 for a in actions[s]} for s in non_terminal}

    qvf_iter = find_qvf_sarsa_control(
        states=states,
        non_terminal_states=non_terminal,
        actions=actions,
        q_0=q_0,
        transitions=transition_function,
        γ=0.9
    )

    for qvf in qvf_iter:
        for s in qvf:
            print('{}: {}'.format(s, qvf[s]))
        print()
