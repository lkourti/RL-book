from dataclasses import dataclass
from typing import Mapping, Dict, Sequence, Tuple
from rl.distribution import Categorical
from rl.markov_process import Transition, FiniteMarkovProcess
from rl.gen_utils.plot_funcs import plot_list_of_curves
import collections
import numpy as np
import itertools

@dataclass(frozen=True)
class BlockState:
    block: int


SLMapping = Dict[int, int]

class SnakesLaddersFMP(FiniteMarkovProcess[BlockState]):
    board_size: int
    dice_size: int
    sl_mapping: SLMapping

    def __init__(self, sl_mapping: SLMapping):
        self.board_size = 100
        self.dice_size = 6
        self.sl_mapping = sl_mapping
        FiniteMarkovProcess.__init__(self, self.get_transition_map())

    def get_transition_map(self) -> Transition[BlockState]:
        d: Dict[BlockState, Categorical[BlockState]] = {}
        for i in range(self.board_size - self.dice_size + 1):
            state_probs_map: Mapping[BlockState, float] =\
                {BlockState(self.sl_mapping[j]): 1/self.dice_size for j in range(i+1, i+self.dice_size+1)}
            d[BlockState(i)] = Categorical(state_probs_map)
        for i in range(self.board_size - self.dice_size + 1, self.board_size):
            state_probs_map: Mapping[BlockState, float] =\
                {BlockState(self.sl_mapping[j]): 1/self.dice_size for j in range(i+1, self.board_size+1)}
            state_probs_map[BlockState(i)] = (self.dice_size - (self.board_size-i)) / self.dice_size
            d[BlockState(i)] = Categorical(state_probs_map)
        d[BlockState(self.board_size)] = None
        return d

    def generate_traces(self, num_traces: int) -> np.ndarray:
        s0_dist_dict: Mapping[BlockState, float] = {BlockState(0): 1}
        s0_dist = Categorical(s0_dist_dict)
        all_traces = []
        for trace in itertools.islice(self.traces(s0_dist), num_traces):
            one_trace = []
            for s in trace:
                one_trace.append(s.block)
            all_traces.append(np.array(one_trace))
        return np.array(all_traces)

    def find_expected_num_steps(self) -> float:
        transition_matrix = self.get_transition_matrix()
        q = np.identity(self.board_size) - transition_matrix
        ones = np.ones(self.board_size)
        t = np.linalg.solve(q, ones)
        return round(t[0], 5)

def get_finish_time_histogram(finish_times: np.ndarray) -> Tuple[Sequence[int], Sequence[int]]:
    count_dict = collections.Counter(finish_times)
    count_dict_sorted = collections.OrderedDict(sorted(count_dict.items()))
    return [x for x, _ in count_dict_sorted.items()], [y for _, y in count_dict_sorted.items()]

def plot_finish_time_distr(finish_times: np.ndarray, num_traces: int):
    x_finish_time, y_counter = get_finish_time_histogram(finish_times)
    plot_list_of_curves(
        [x_finish_time],
        [y_counter],
        ["b"],
        [f"Finish Time Distribution (traces={num_traces:d})"],
        "Finish Time",
        "Counts",
        r"Finish Time Distribution"
    )

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

    #The lines below verify the result we get in prob4.py for the expected number of steps
    #expected_num_steps = sl_mp.find_expected_num_steps()
    #print("Expected number of dice rolls to finish the game: {}".format(expected_num_steps))

    sl_mp = SnakesLaddersFMP(sl_mapping)
    num_traces = 1000
    sl_traces = sl_mp.generate_traces(num_traces)
    finish_times = np.array([len(trace) for trace in sl_traces])
    plot_finish_time_distr(finish_times, num_traces)

