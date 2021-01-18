from dataclasses import dataclass
from numpy.random import binomial
from typing import Callable, Tuple, Optional
from rl.distribution import SampledDistribution, Categorical
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
from rl.markov_process import MarkovRewardProcess



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


if __name__ == '__main__':
    start_price: int = 100
    start_state: PriceState = PriceState(start_price)
    level_param: int = 100
    alpha1: float = 0.25
    def f(state: PriceState) -> float:
        return 0.1*state.price
    stock_price_mrp = StockPriceMRP(level_param, alpha1, f)

    #s = start_state
    #sampler = stock_price_mrp.transition_reward(s)
    #s,r = sampler.sample()
    #print(s)
    #print(r)

