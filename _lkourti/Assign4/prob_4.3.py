from typing import Sequence, Callable, Mapping, Tuple, Dict
from math import log

class WageMDP:
    def __init__(self, wages:Sequence[float], probs:Sequence[float], alpha:float, gamma:float):
        self.wages = wages
        self.probs = probs
        self.alpha = alpha
        self.gamma = gamma
        self.utils = self.utilities(wages)

    def utility_fun(self, x:float) -> float:
        return log(x)

    def utilities(self, wages:Sequence) -> Sequence[float]:
        utils = [self.utility_fun(w) for w in wages]
        return utils

    def get_opt_vf(self) -> Mapping[Tuple[int,str],float]:
        n: int = len(self.probs)
        u = 'U'
        e = 'E'
        vf: Dict[Tuple[int,str], float] = {(i,'U'):0 for i in range(1,n+1)}
        vf.update({(i,'E'): 0 for i in range(1,n+1)})
        tol = 1e-6
        epsilon = tol * 1e6
        while epsilon >= tol:
            old_vf = vf.copy()
            for i in range(1,n+1):
                vf[(i,u)] = max(self.utils[0] + self.gamma * sum(self.probs[j-1]*old_vf[(j,u)] for j in range(1,n+1)),
                                old_vf[(i,e)])
                vf[(i,e)] = self.utils[i] + \
                            self.gamma * ((1-self.alpha)*old_vf[(i,e)] +
                                          self.alpha * sum(self.probs[j-1]*old_vf[(j,u)] for j in range(1,n+1)))

                epsilon = max(abs(old_vf[state] - vf[state]) for state in vf)
        return vf

    def get_opt_policy(self) -> Mapping[int,str]:
        vf:Mapping[Tuple[int,str],float] = self.get_opt_vf()
        policy: Mapping[int, str] = {}
        n: int = len(self.probs)
        for state in vf:
            job = state[0]
            status = state[1]
            if status == 'U':
                decline_reward = self.utils[0] + self.gamma * sum(self.probs[j-1]*vf[(j,'U')] for j in range(1,n+1))
                accept_reward = vf[(job,'E')]
                if accept_reward >= decline_reward:
                    policy[job] = 'Accept'
                else:
                    policy[job] = 'Decline'
        return policy

    def pprint_vf(self):
        vf: Mapping[Tuple[int, str], float] = self.get_opt_vf()
        print("Expected reward if the worker starts:")
        reward = 0
        for state in vf:
            job = state[0]
            status = state[1]
            if status == 'U':
                reward += vf[state] * self.probs[job-1]
        print("  Unemployed: {}".format(round(reward,3)))
        for state in vf:
            job = state[0]
            status = state[1]
            if status == 'U':
                print("    with offer for job {}: {}".format(job, round(vf[state],3)))
            else:
                print("  Employed at job {}: {}".format(job, round(vf[state],3)))
        print(" ")

    def pprint_policy(self):
        policy: Mapping[int,str] = self.get_opt_policy()
        print("Optimal actions in case of unemployment:")
        for job in policy:
            print("  Offer for job {}: {}".format(job, policy[job]))
        print(" ")

if __name__ == '__main__':
    probs: Sequence[float] = [0.6, 0.3, 0.1]
    wages: Sequence[float] = [1, 1.5, 2, 10]
    gamma: float = 0.9
    alpha: float = 0.15
    wage_mdp = WageMDP(wages, probs, alpha, gamma)
    opt_vf = wage_mdp.get_opt_vf()
    opt_policy = wage_mdp.get_opt_policy()
    wage_mdp.pprint_vf()
    wage_mdp.pprint_policy()