from rl.chapter7.asset_alloc_discrete import AssetAllocDiscrete
from rl.function_approx import DNNSpec
from typing import Sequence, Callable, Tuple, TypeVar, Iterable, Optional
from rl.distribution import Gaussian, Distribution, Categorical
from rl.returns import returns
from rl.markov_decision_process import TransitionStep
from rl.markov_decision_process import Policy
import numpy as np

S = TypeVar('S')
A = TypeVar('A')


def reinforce(
        num_episodes: int,
        features_funcs: Sequence[Callable[[Tuple[S, A]], float]],
        actions: Callable[[S], Iterable[A]],
        init_wealth_distr: Gaussian,
        get_episode: Callable[[Distribution[S], Policy[S, A]], Iterable[TransitionStep[S, A]]],
        γ: float,
        alpha: float,
        # softmax: bool = True
):
    def get_phi_sa(s: S, a: A):
        return np.array([f((s, a)) for f in features_funcs])

    class SoftMaxPolicy(Policy[S, A]):
        def __init__(self, theta: np.ndarray):
            self.theta = theta

        def act(self, s: S) -> Optional[Distribution[A]]:
            probs_dict = {}
            for a in actions(s):
                numerator = np.exp(np.dot(get_phi_sa(s, a), theta))
                if numerator < 0.0001:
                    continue
                denominator = np.sum([np.exp(np.dot(get_phi_sa(s, b), self.theta)) for b in actions(s)
                                      if np.exp(np.dot(get_phi_sa(s, b), self.theta)) >= 0.0001])
                probs_dict[a] = numerator / denominator
            return Categorical(probs_dict)

    num_features = len(features_funcs)
    theta = np.zeros(num_features)
    for k in range(num_episodes):
        ep = get_episode(init_wealth_distr, SoftMaxPolicy(theta))
        episode = list(returns(ep, γ, γ**30))
        for t in range(len(episode)):
            s = episode[t].state
            a = episode[t].action
            phi_sa = get_phi_sa(s, a)
            normalization = sum([np.exp(np.dot(get_phi_sa(s, b), theta)) for b in actions(s)
                                if np.exp(np.dot(get_phi_sa(s, b), theta)) >= 0.0001])
            sum_pi = sum([np.exp(np.dot(get_phi_sa(s, b), theta)) * get_phi_sa(s, b) for b in actions(s)
                          if np.exp(np.dot(get_phi_sa(s, b), theta)) >= 0.0001])
            derivative = phi_sa - sum_pi / normalization
            theta += alpha * γ**t * derivative * episode[t].return_
    return theta


if __name__ == '__main__':
    steps: int = 4
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_var: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        print(f"Time {t:d}")
        print()
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        val: float = - np.exp(- excess * excess * left / (2 * var)
                              - a * growth * (1 + r) * init_wealth) / a
        bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
            np.log(np.abs(a))
        w_t_wt: float = a * growth * (1 + r)
        x_t_wt: float = a * excess * growth
        x_t2_wt: float = - var * (a * growth) ** 2 / 2

        print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {val:.3f}")
        print(f"Bias Weight = {bias_wt:.3f}")
        print(f"W_t Weight = {w_t_wt:.3f}")
        print(f"x_t Weight = {x_t_wt:.3f}")
        print(f"x_t^2 Weight = {x_t2_wt:.3f}")
        print()

    # MDP
    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_var)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    aad_mdp = aad.get_mdp(3)
    num_episodes = 100
    reinf_policy = reinforce(num_episodes, feature_funcs, aad_mdp.actions,
                             init_wealth_distr, aad_mdp.simulate_actions, γ=0.9, alpha=0.01)
