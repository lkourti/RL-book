from dataclasses import dataclass
from typing import Dict, Tuple, Sequence
from rl.iterate import converge
from rl.dynamic_programming import policy_iteration_result, value_iteration_result
from rl.distribution import Categorical, Constant
from rl.markov_decision_process import  StateActionMapping, FinitePolicy, FiniteMarkovDecisionProcess
from rl.markov_process import FiniteMarkovRewardProcess
import matplotlib.pyplot as plt
from pprint import pprint
import time
import numpy as np


@dataclass(frozen=True)
class LilypadState:
    lilypad: int

LilypadSoundMapping = StateActionMapping[LilypadState, str]

class FrogEscapeMDP(FiniteMarkovDecisionProcess[LilypadState, str]):
    def __init__(self, num_lilypads:int):
        self.num_lilypads = num_lilypads
        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> LilypadSoundMapping:
        d: Dict[LilypadState, Dict[str, Categorical[Tuple[LilypadState,
                                                            float]]]] = {}
        for lilypad in range(1,self.num_lilypads-1):
            state: LilypadState = LilypadState(lilypad)
            d1: Dict[str,Categorical[Tuple[LilypadState, float]]] = {}

            # sound A
            sr_probs_dict_a: Dict[Tuple[LilypadState, float], float] = {}
            sr_probs_dict_a[(LilypadState(lilypad - 1), 0)] = lilypad / (self.num_lilypads-1)
            if lilypad + 1 == self.num_lilypads-1:
                sr_probs_dict_a[(LilypadState(lilypad + 1), 1)] = 1 - lilypad/(self.num_lilypads-1)
            else:
                sr_probs_dict_a[(LilypadState(lilypad + 1), 0)] = 1 - lilypad /(self.num_lilypads-1)
            d1['A'] = Categorical(sr_probs_dict_a)

            # sound B
            sr_probs_dict_b: Dict[Tuple[LilypadState, float], float] = {}
            for i in range(self.num_lilypads):
                if i != lilypad:
                    if i == self.num_lilypads - 1:
                        sr_probs_dict_b[(LilypadState(i), 1)] = 1/(self.num_lilypads-1)
                    else:
                        sr_probs_dict_b[(LilypadState(i), 0)] = 1 /(self.num_lilypads-1)
            d1['B'] = Categorical(sr_probs_dict_b)

            d[state] = d1
        return d

    def get_all_action_combinations(self) -> list:
        n = self.num_lilypads - 2
        s = ''
        for _ in range(n):
            s += '1'
        tot_num = int(s,2)
        all_policies = []
        for i in range(tot_num+1):
            all_policies.append(bin(i)[2:].zfill(n))
        return all_policies


    def get_all_deterministic_policies(self) -> Sequence[FinitePolicy[LilypadState, str]]:
        bin_to_act = {'0':'A', '1':'B'}
        all_action_comb = self.get_all_action_combinations()
        all_policies = []
        for action_comb in all_action_comb:
            policy: FinitePolicy[LilypadState,str] = FinitePolicy(
                {LilypadState(i+1): Constant(bin_to_act[a]) for i,a in enumerate(action_comb)}
            )
            all_policies.append(policy)
        return all_policies


    def get_optimal(self, policies: Sequence[FinitePolicy[LilypadState, str]]) -> \
            Tuple[FinitePolicy[LilypadState, str], np.ndarray]:

        gamma = 1
        optimal_policy: FinitePolicy[LilypadState, str] = policies[0]
        optimal_value_fun: np.ndarray = self.apply_finite_policy(policies[0]).get_value_function_vec(gamma)
        for policy in policies:
            implied_mrp: FiniteMarkovRewardProcess[LilypadState] = self.apply_finite_policy(policy)
            value_fun: np.ndarray = implied_mrp.get_value_function_vec(gamma)
            if (value_fun >= optimal_value_fun).all():
                optimal_policy = policy
                optimal_value_fun = value_fun
        return (optimal_policy, optimal_value_fun)



if __name__ == '__main__':
    x = range(4, 17)
    y_brute = []
    y_pi = []
    y_vi = []
    for number_of_lilypads in x:
        frog_mdp: FrogEscapeMDP = FrogEscapeMDP(number_of_lilypads)

        all_det_policies: Sequence[FinitePolicy[LilypadState, str]] = frog_mdp.get_all_deterministic_policies()
        t1 = time.time()
        optimal_det_policy, optimal_value_fun = frog_mdp.get_optimal(all_det_policies)
        t2 = time.time()
        time_brute_force = t2 - t1
        y_brute.append(time_brute_force)

        # Policy Iteration
        t1 = time.time()
        opt_vf_pi, opt_policy_pi = policy_iteration_result(frog_mdp, gamma=1)
        t2 = time.time()
        time_policy_iter = t2 - t1
        y_pi.append(time_policy_iter)
        #pprint(opt_vf_pi)
        #print(opt_policy_pi)

        # Value Iteration
        t1 = time.time()
        opt_vf_pi, opt_policy_pi = value_iteration_result(frog_mdp, gamma=1)
        t2 = time.time()
        time_value_iter = t2 - t1
        y_vi.append(time_value_iter)
        #pprint(opt_vf_pi)
        #print(opt_policy_pi)

    plt.plot(x, y_brute, c='r', label='Brute Force')
    plt.plot(x, y_pi, c='b', label='Policy Iteration')
    plt.plot(x, y_vi, c='g', label='Value Iteration')
    plt.xlabel('Number of Lilypads')
    plt.ylabel('Time till Convergence')
    plt.legend()
    plt.show()



