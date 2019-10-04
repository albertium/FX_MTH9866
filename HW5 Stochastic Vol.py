from scipy.integrate import quad
from scipy.stats import norm
import numpy as np


def price_BS(spot, fp, K, r, sig, tau):
    F = spot + fp
    d1 = (np.log(F / K) + (sig ** 2 / 2) * tau) / sig / np.sqrt(tau)
    d2 = d1 - sig * np.sqrt(tau)
    price = F * norm.cdf(d1) - K * norm.cdf(d2)
    return price * np.exp(-r * tau)


def price_with_conditional_BS(spot, fp, K, r, sig, tau, j_mu, j_sig, lmbda):
    pass


def price_with_characteristic(spot, fp, K, r, tau, f):
    """
    fp is the forward point
    """
    ratio = np.log(K / spot)

    def integrand(theta):
        return f(theta) * np.exp(-1j * theta * ratio) / (theta ** 2 + 1j * theta)

    price = spot + fp - K / 2 - K / np.pi * quad(integrand, 0, np.inf)[0]
    return price * np.exp(-tau * r)


def get_characteristic(mu, tau, sigma, j_mu, j_sig, lmbda):
    sig2 = sigma ** 2

    def f(theta):
        theta2 = theta ** 2
        return 1j * theta * mu * tau - theta2 * sig2 * tau / 2 \
               + lmbda * tau * (np.exp(1j * theta * j_mu - theta2 * j_sig / 2) - 1)

    return f


if __name__ == '__main__':
    spot = 1
    tau = 0.5
    fp = 0.03
    r = 0.05
    sig = 0.07
    j_mu = -0.04
    j_sig = 0.15 ** 2
    lmbda = 3

    character = get_characteristic(0.01, tau, sig, j_mu, j_sig, lmbda)
    res = price_with_characteristic(spot, fp, spot, r, tau, character)
    print(res)