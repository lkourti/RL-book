'''
It is observed that all the algorithms return the same Value Function
apart from the MRP value function. This stems from the fact that the
MRP value function is using the empirical probabilities and reward
functions in place of the true ones.
'''
from typing import Sequence, Tuple, Mapping, Dict
from operator import itemgetter
import numpy as np
from itertools import groupby
from numpy.random import randint

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    vf: Dict[S, float] = {s[0]: 0 for s in state_return_samples}
    freq_s: Dict[S, float] = {s[0]: 0 for s in state_return_samples}
    for pair in state_return_samples:
        vf[pair[0]] += pair[1]
        freq_s[pair[0]] += 1
    vf.update({s: vf[s]/freq_s[s] for s in vf})
    return vf


def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    s_to_rs: Dict[S, Sequence[Tuple[float, S]]] = \
        {s: [(r, s1) for _, r, s1 in srs] for s, srs in
         groupby(sorted(srs_samples, key=itemgetter(0)), itemgetter(0))}
    prob_func: ProbFunc = {s: {s1: len(list(rs1_list)) / len(list(rs_list))
                               for s1, rs1_list in groupby(sorted(rs_list, key=itemgetter(1)), itemgetter(1))
                               if s1 != 'T'}
                           for s, rs_list in s_to_rs.items()}
    reward_func = {s: np.mean([r for r, _ in rs_list]) for s, rs_list in s_to_rs.items()}
    return prob_func, reward_func


def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    states = list(prob_func.keys())
    rewards = np.array([reward_func[s] for s in states])
    prob = np.array([[prob_func[s][s1] if s1 in prob_func[s] else 0
                      for s1 in states] for s in states])
    vf = np.linalg.inv(np.eye(len(states)) - prob).dot(rewards)
    return {states[i]: vf[i] for i in range(len(states))}


def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    vf = {s: [0.] for s in set(s0 for s0, _, _ in srs_samples)}
    num_samples = len(srs_samples)
    for updates in range(num_updates):
        alpha = learning_rate * (updates / learning_rate_decay + 1) ** -0.5
        s, r, s1 = srs_samples[randint(num_samples, size=1)[0]]
        vf[s].append(vf[s][-1] + alpha * (r + (vf[s1][-1] if s1 != 'T' else 0.) - vf[s][-1]))
    return {s: np.mean(v[-int(len(v) * 0.9):]) for s, v in vf.items()}


def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    states = list(set(s for s, _, _ in srs_samples))
    num_states = len(states)
    phi = np.eye(num_states)
    A = np.zeros((num_states, num_states))
    b = np.zeros(num_states)
    for s, r, s1 in srs_samples:
        p = phi[states.index(s1)] if s1 != 'T' else np.zeros(num_states)
        A += np.outer(phi[states.index(s)], phi[states.index(s)] - p)
        b += phi[states.index(s)] * r
    return {states[i]: v for i, v in enumerate(np.linalg.inv(A).dot(b))}


if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))