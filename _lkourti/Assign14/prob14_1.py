from typing import Iterable, TypeVar, Callable, Sequence
from rl.markov_process import TransitionStep
import numpy as np

S = TypeVar('S')
A = TypeVar('A')


class LSTD:
    def __init__(self, features: Sequence[Callable[[S], float]], γ: float):
        self.features: Sequence[Callable[[S], float]] = features
        self.γ: float = γ
        self.m: int = len(features)
        self.A: np.ndarray = np.zeros((self.m, self.m))
        self.b: np.ndarray = np.zeros(self.m)

    def update(self, data: Iterable[TransitionStep[S]]):
        for step in data:
            state: S = step.state
            reward: float = step.reward
            state1: S = step.next_state
            features_s: np.ndarray = np.array([f[state] for f in self.features])
            features_s1: np.ndarray = np.array([f[state1] for f in self.features])
            self.A += np.outer(features_s, features_s - self.γ * features_s1)
            self.b += reward * features_s

    def solve(self) -> np.ndarray:
        weights: np.ndarray = np.dot(np.linalg.pinv(self.A), self.b)
        return weights
