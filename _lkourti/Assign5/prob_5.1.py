from __future__ import annotations
from dataclasses import dataclass, replace, field
from operator import itemgetter
from scipy.interpolate import splrep, BSpline
from typing import Callable, Iterable, Sequence, Tuple, TypeVar, Optional
from rl.function_approx import FunctionApprox
import matplotlib.pyplot as plt
import numpy as np

X = TypeVar('X')

@dataclass(frozen=True)
class BSplineApprox(FunctionApprox[X]):
    feature_function: Callable[[X], float]
    degree: int
    knots: np.ndarray = field(default_factory=lambda: np.array([]))
    coeffs: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_feature_values(self, input_x_seq: Iterable[X]) -> Sequence[float]:
        return [self.feature_function(x) for x in input_x_seq]

    def representational_gradient(self, x: X) -> BSplineApprox[X]:
        # nothing useful
        return self

    def evaluate(self, input_x_seq: Iterable[X]) -> np.ndarray:
        features: Sequence[float] = self.get_feature_values(input_x_seq)
        spl: BSpline = BSpline(self.knots, self.coeffs, self.degree)
        return spl(features)

    def update(self, xy_input_seq: Iterable[Tuple[X, float]]) -> BSplineApprox[X]:
        x_input_seq, y_input_seq = zip(*xy_input_seq)
        feature_vals: Sequence[float] = self.get_feature_values(x_input_seq)
        sorted_pairs: Sequence[Tuple[float, float]] = \
            sorted(zip(feature_vals, y_input_seq), key=itemgetter(0))
        new_knots, new_coeffs, _ = splrep(
            [feature for feature, _ in sorted_pairs],
            [y for _, y in sorted_pairs],
            k=self.degree
        )
        return replace(self, knots=new_knots, coeffs=new_coeffs)

    def solve(self, xy_input_seq: Iterable[Tuple[X, float]], error_tolerance: Optional[float] = None) \
            -> BSplineApprox[X]:
        return self.update(xy_input_seq)

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        if isinstance(other, BSplineApprox):
            flag_1: bool = np.all(np.abs(self.knots - other.knots) <= tolerance)
            flag_2: bool = np.all(np.abs(self.coeffs - other.coeffs) <= tolerance)
            return flag_1 and flag_2
        return False



if __name__ == '__main__':
    xx = np.linspace(0, 10, 10)
    yy = np.sin(xx)
    xy = tuple(zip(xx, yy))

    degree = 3
    bspline_approx = BSplineApprox(lambda x: x, degree)
    approx = bspline_approx.solve(xy)

    xx_test = np.linspace(0, 10, 100)
    plt.scatter(xx, yy, c='b', s=12, label='Input Data')
    plt.plot(xx_test, approx.evaluate(xx_test), c='r', label='B-Spline')
    plt.legend()
    plt.show()
