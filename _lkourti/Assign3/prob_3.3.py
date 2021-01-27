from dataclasses import dataclass
from typing import Dict, Tuple, Sequence
from rl.distribution import Categorical, Constant
from rl.markov_decision_process import  StateActionMapping, FinitePolicy, FiniteMarkovDecisionProcess
from rl.markov_process import FiniteMarkovRewardProcess
import numpy as np
import matplotlib.pyplot as plt


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
    number_of_lilypads: int = 13
    frog_mdp: FrogEscapeMDP = FrogEscapeMDP(number_of_lilypads)
    all_det_policies: Sequence[FinitePolicy[LilypadState, str]] = frog_mdp.get_all_deterministic_policies()
    optimal_det_policy, optimal_value_fun = frog_mdp.get_optimal(all_det_policies)
    implied_mrp = frog_mdp.apply_finite_policy(optimal_det_policy)
    optimal_q_fun = implied_mrp.reward_function_vec
    print("Optimal Value Function: {}".format(optimal_value_fun))
    print("Optimal Policy:")
    print(optimal_det_policy)

    croak_color = []
    for i in range(1,number_of_lilypads-1):
        action_prob = optimal_det_policy.act(LilypadState(i))
        action = action_prob.sample()
        if action == 'A':
            croak_color.append('r')
        else:
            croak_color.append('b')

    plt.scatter(range(number_of_lilypads-2), optimal_value_fun, c=croak_color, s=20, marker='o')
    plt.title('n = '+str(number_of_lilypads-1))
    plt.xlabel("Lilypad Number")
    plt.ylabel("Probability of Escape")
    #plt.legend()
    plt.show()

