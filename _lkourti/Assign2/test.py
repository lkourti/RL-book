from dataclasses import dataclass
from typing import Mapping, Dict
from rl.distribution import Categorical
from rl.markov_process import Transition, FiniteMarkovProcess
from scipy.stats import poisson
import numpy as np

x: Mapping[int,int]
x = {i:i for i in range(5)}
x[2]=100
print(x)