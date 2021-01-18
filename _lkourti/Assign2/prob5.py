from dataclasses import dataclass
from numpy.random import binomial
from typing import Callable, Tuple, Mapping
from rl.distribution import SampledDistribution, Categorical
from rl.gen_utils.common_funcs import get_logistic_func
from rl.markov_process import MarkovRewardProcess
import itertools
import numpy as np


@dataclass(frozen=True)
class PriceState:
    price: int

class StockPriceMRP(MarkovRewardProcess[PriceState]):
    level_param: int            # level to which price mean-reverts
    alpha1: float = 0.25        # strength of mean-reversion (non-negative value)
    reward_function: Callable   # reward function f

    def __init__(self, level_param: int, alpha1: float, reward_function: Callable):
        self.level_param = level_param
        self.alpha1 = alpha1
        self.reward_function = reward_function

    def transition_reward(self, state: PriceState) -> SampledDistribution[Tuple[PriceState, float]]:

        def sample_next_state_reward(state=state) -> Tuple[PriceState, float]:
            up_prob = get_logistic_func(self.alpha1)(self.level_param - state.price)
            up_move: int = binomial(1, up_prob, 1)[0]
            next_state: PriceState = PriceState(price=state.price + up_move * 2 - 1)
            reward: float = self.reward_function(next_state)
            return next_state, reward

        return SampledDistribution(sample_next_state_reward)

    def generate_traces(self, start_state: PriceState, trace_length: int, num_traces: int) -> (np.ndarray, np.ndarray):
        s0_dist_dict: Mapping[PriceState, float] = {start_state: 1}
        s0_dist = Categorical(s0_dist_dict)
        all_traces_state_price = []
        all_traces_rewards = []
        for trace_counter in range(num_traces):
            one_trace_state_price = [start_state.price]
            one_trace_reward = []
            for transition_step in itertools.islice(self.simulate_reward(s0_dist), trace_length):
                one_trace_state_price.append(transition_step.next_state)
                one_trace_reward.append(transition_step.reward)
            all_traces_state_price.append(one_trace_state_price)
            all_traces_rewards.append(one_trace_reward)
        return np.array(all_traces_state_price), np.array(all_traces_rewards)

    def compute_returns(self, reward_traces: np.ndarray, gamma: float) -> float:
        num_traces = reward_traces.shape[0]
        returns = []
        for i in range(num_traces):
            return_g = 0
            one_reward_trace = reward_traces[i,:]
            for k, reward in enumerate(one_reward_trace):
                return_g += gamma**k * reward
            returns.append(return_g)
        return returns



if __name__ == '__main__':
    start_price: int = 100
    start_state: PriceState = PriceState(start_price)
    level_param: int = 100
    alpha1: float = 0.25
    gamma = 0.8
    num_traces = 100
    trace_length = 1000
    def f(state: PriceState) -> float:
        return 0.1*state.price
    stock_price_mrp = StockPriceMRP(level_param, alpha1, f)

    stock_price_traces, reward_traces = stock_price_mrp.generate_traces(start_state, trace_length, num_traces)
    traces_returns = stock_price_mrp.compute_returns(reward_traces, 0.8)
    value_function_approx = np.mean(traces_returns)
    print("Value Function for starting price {}: {}".format(start_price, round(value_function_approx,3)))