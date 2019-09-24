
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class FXVolSmile:
    def __init__(self, spot, ff,  r, t, atm, rr25, rr10, bf25, bf10, F):
        self.spot = spot
        self.ff = ff
        self.r = r
        self.t = t
        self.q = r - np.log((spot + ff) / spot) / t  # solve for asset interest rate

        # solve implied volatilities
        c25, p25 = self.solve_vols(atm, rr25, bf25)
        c10, p10 = self.solve_vols(atm, rr10, bf10)
        self.c10 = c10
        self.p10 = p10
        self.ys = np.array([p10, p25, atm, c25, c10])

        # solve strikes
        Katm = (spot + ff) * np.exp(atm ** 2 * t / 2)
        Kc25 = self.solve_strikes(c25, 0.25)
        Kp25 = self.solve_strikes(p25, -0.25)
        Kc10 = self.solve_strikes(c10, 0.1)
        Kp10 = self.solve_strikes(p10, -0.1)
        Kmin = Kp10 * np.exp(-F * p10 * np.sqrt(t))
        Kmax = Kc10 * np.exp(F * c10 * np.sqrt(t))

        self.xs = np.array([Kmin, Kp10, Kp25, Katm, Kc25, Kc10, Kmax])
        self.N = len(self.xs) - 1
        self.coefs = self.get_spline_coefficients()

    @staticmethod
    def solve_vols(atm, rr, bf):
        """
        solve for the wing vols given risk reversal and butterfly
        """
        c = bf + atm + rr / 2
        p = c - rr
        return c, p

    def solve_strikes(self, sig, delta):
        """
        solve for strikes given spot, forward point, asset interest rate, maturity and delta
        """
        sign = 1 if delta > 0 else -1
        Q = sig ** 2 / 2 * self.t - sign * sig * np.sqrt(self.t) * norm.ppf(sign * delta * np.exp(self.q * self.t))
        return (self.spot + self.ff) * np.exp(Q)

    def get_spline_coefficients(self):
        # for solving for coefficients with Ax = b
        A = np.zeros((self.N * 4, self.N * 4))
        b = np.zeros(self.N * 4)

        x_basis = np.array([np.power(self.xs, p) for p in [0, 1, 2, 3]]).T
        for i in range(self.N - 1):
            xs = x_basis[i + 1]
            y = self.ys[i]
            _, x, x2, _ = xs
            # conditions for f(xi) = yi, f(x(i+1)) = y(i+1)
            idx = i * 4
            A[idx, idx + 4: idx + 8] = xs
            A[idx + 1, idx: idx + 4] = xs
            b[idx] = b[idx + 1] = y

            # conditions for f'(xi-1) == f'(xi)
            A[idx + 2, idx + 1: idx + 4] = (1, 2 * x, 3 * x2)
            A[idx + 2, idx + 5: idx + 8] = (-1, -2 * x, -3 * x2)

            # conditions for f''(xi-1) == f''(xi)
            A[idx + 3, idx + 2: idx + 4] = (2, 6 * x)
            A[idx + 3, idx + 6: idx + 8] = (-2, -6 * x)

        # boundary conditions
        # f'(x0) = f''(x0) = 0
        _, x, x2, _ = x_basis[0]
        A[-4, 1: 4] = (1, 2 * x, 3 * x2)
        A[-3, 2: 4] = (2, 6 * x)

        # f'(xN) = f''(xN) = 0
        _, x, x2, _ = x_basis[-1]
        A[-2, -3:] = (1, 2 * x, 3 * x2)
        A[-1, -2:] = (2, 6 * x)

        coefs = np.linalg.solve(A, b)
        return coefs.reshape(-1, 4)

    def volatility(self, x):
        x = np.minimum(np.maximum(x, self.xs[0]), self.xs[-1])  # vol is constant outside the boundary
        idx = np.maximum(np.searchsorted(self.xs, x) - 1, 0)
        coef = self.coefs[idx]
        return np.sum(coef * np.array([np.power(x, p) for p in [0, 1, 2, 3]]).T, axis=1)


if __name__ == '__main__':
    spot = 1
    forward_point = 0
    r = 0
    t = 0.5

    smile = FXVolSmile(spot, forward_point, r, t, 0.08, 0.01, 0.018, 0.0025, 0.008, 0.01)
    smile2 = FXVolSmile(spot, forward_point, r, t, 0.08, 0.01, 0.018, 0.0025, 0.008, 10)
    K_min, K_max = smile.solve_strikes(smile.p10, -0.01), smile.solve_strikes(smile.c10, 0.01)
    fig = plt.figure()
    ax = plt.axes()
    xs = np.linspace(K_min, K_max, 1000)
    ax.plot(xs, smile.volatility(xs))
    ax.plot(xs, smile2.volatility(xs))
    plt.show()