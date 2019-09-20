
import numpy as np
from scipy import sparse


class Spline:
    def __init__(self, ys, xs):
        xs, ys = np.array(list(zip(*[[y, x] for y, x in sorted(zip(xs, ys))])))  # sort by x's
        self.coefs = self.get_spline_coefficients(ys, xs)
        self.xs = xs

    @staticmethod
    def get_spline_coefficients(ys, xs):
        diff = xs[1:] - xs[:-1]
        diags = [diff[1:-1], 2 * (xs[2:] - xs[:-2]), diff[1:-1]]
        M = sparse.diags(diags, [-1, 0, 1]).toarray()
        slope = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        z = 6 * (slope[1:] - slope[:-1])
        w = np.zeros(len(xs))
        w[1:-1] = np.linalg.solve(M, z)
        diff = xs[1:] - xs[:-1]
        c = (w[:-1] * xs[1:] - w[1:] * xs[:-1]) / 2 / diff
        d = (w[1:] - w[:-1]) / 6 / diff
        x_sq, x_cub = np.power(xs, 2), np.power(xs, 3)
        q = ys[:-1] - c * x_sq[:-1] - d * x_cub[:-1]
        r = ys[1:] - c * x_sq[1:] - d * x_cub[1:]
        a = (q * xs[1:] - r * xs[:-1]) / diff
        b = (r - q) / diff
        return np.vstack([a, b, c, d]).T

    def evaluate(self, x):
        coef = self.coefs[np.maximum(np.searchsorted(self.xs, x) - 1, 0)]
        return np.sum(coef * np.array([np.power(x, p) for p in [0, 1, 2, 3]]), axis=1)


spline = Spline([2, 3, 4, 5], [1, 2, 4, 8])
print(spline.evaluate([1, 2, 4, 8]))
