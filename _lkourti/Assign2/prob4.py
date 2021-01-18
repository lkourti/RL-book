from typing import Mapping, Dict, Tuple
from prob2 import BlockState, SLMapping, SnakesLaddersFMP
from rl.distribution import Categorical
from rl.markov_process import RewardTransition, FiniteMarkovRewardProcess

class SnakesLaddersFMRP(SnakesLaddersFMP, FiniteMarkovRewardProcess[BlockState]):
    def __init__(self, sl_mapping: SLMapping):
        SnakesLaddersFMP.__init__(self, sl_mapping)
        FiniteMarkovRewardProcess.__init__(self, self.get_transition_reward_map())


    def get_transition_reward_map(self) -> RewardTransition[BlockState]:
        d: Dict[BlockState, Categorical[Tuple[BlockState, float]]] = {}
        for i in range(self.board_size - self.dice_size + 1):
            sr_probs_map: Mapping[Tuple[BlockState, float], float] = {}
            #{(BlockState(self.sl_mapping[j]), 1): 1 / self.dice_size for j in range(i + 1, i + self.dice_size + 1)}
            for j in range(i + 1, i + self.dice_size + 1):
                if (BlockState(self.sl_mapping[j]), 1) not in sr_probs_map:
                    sr_probs_map[(BlockState(self.sl_mapping[j]), 1)] = 1 / self.dice_size
                else:
                    sr_probs_map[(BlockState(self.sl_mapping[j]), 1)] += 1 / self.dice_size
            d[BlockState(i)] = Categorical(sr_probs_map)
        for i in range(self.board_size - self.dice_size + 1, self.board_size):
            sr_probs_map: Mapping[Tuple[BlockState, float], float] = {}
            #{(BlockState(self.sl_mapping[j]), 1): 1 / self.dice_size for j in range(i + 1, self.board_size + 1)}
            for j in range(i + 1, self.board_size + 1):
                if (BlockState(self.sl_mapping[j]), 1) not in sr_probs_map:
                    sr_probs_map[(BlockState(self.sl_mapping[j]), 1)] = 1 / self.dice_size
                else:
                    sr_probs_map[(BlockState(self.sl_mapping[j]), 1)] += 1 / self.dice_size
            sr_probs_map[(BlockState(i), 1)] = (self.dice_size - (self.board_size - i)) / self.dice_size
            d[BlockState(i)] = Categorical(sr_probs_map)
        d[BlockState(self.board_size)] = None
        return d

if __name__ == '__main__':
    sl_mapping: SLMapping = {i:i for i in range(100+1)}
    sl_mapping[3] = 39
    sl_mapping[7] = 48
    sl_mapping[12] = 51
    sl_mapping[20] = 41
    sl_mapping[25] = 57
    sl_mapping[28] = 35
    sl_mapping[31] = 6
    sl_mapping[38] = 1
    sl_mapping[45] = 74
    sl_mapping[49] = 8
    sl_mapping[53] = 17
    sl_mapping[60] = 85
    sl_mapping[65] = 14
    sl_mapping[67] = 90
    sl_mapping[69] = 92
    sl_mapping[70] = 34
    sl_mapping[76] = 37
    sl_mapping[77] = 83
    sl_mapping[82] = 63
    sl_mapping[88] = 50
    sl_mapping[94] = 42
    sl_mapping[98] = 54


    sl_mrp = SnakesLaddersFMRP(sl_mapping)
    expected_return = round(sl_mrp.get_value_function_vec(1)[0], 5)
    print("Expected number of dice rolls to finish the game: {}".format(expected_return))