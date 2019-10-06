from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import newton
import numpy as np
import matplotlib.pyplot as plt


def solve_mu(spot, fp, sig, tau, ja, jv, lmbda):
    F = spot + fp
    return np.log(F / spot) / tau - 0.5 * sig ** 2 - lmbda * (np.exp(ja + 0.5 * jv) - 1)


def price_black(F, K, r, sig_total):
    d1 = np.log(F / K) / sig_total + 0.5 * sig_total
    d2 = d1 - sig_total
    price = F * norm.cdf(d1) - K * norm.cdf(d2)
    return price * np.exp(-r * tau)


def price_conditional_black(spot, mu, K, sig, tau, ja, jv, n):
    F = spot * np.exp(mu * tau + 0.5 * sig ** 2 * tau + n * (ja + 0.5 * jv))
    sig_total = np.sqrt(sig ** 2 * tau + n * jv)
    return price_black(F, K, r, sig_total)


def price_with_black(spot, mu, K, sig, tau, ja, jv, lmbda):
    jumps = np.arange(31)
    jumps[0] = 1  # for calculate factorials
    factorials = np.cumprod(jumps)
    jumps[0] = 0
    probs = (lmbda * tau) ** jumps / factorials * np.exp(-lmbda * tau)
    return np.dot(probs, [price_conditional_black(spot, mu, K, sig, tau, ja, jv, n) for n in jumps])


def price_with_characteristic(spot, mu, K, sig, tau, ja, jv, lmbda):
    sig2 = sig ** 2
    x = np.log(spot)

    def phi(theta):
        theta2 = theta ** 2
        at = 1j * theta * mu * tau - 0.5 * theta2 * sig2 * tau \
             + lmbda * tau * (np.exp(1j * theta * ja - theta2 * jv / 2) - 1)
        return np.exp(1j * theta * x + at)

    ratio = np.log(K / spot)

    def integrand(theta):
        return (phi(theta) * np.exp(-1j * theta * ratio) / (theta ** 2 + 1j * theta)).real

    price = spot + fp - K / 2 - K / np.pi * quad(integrand, 0, np.inf)[0]
    return price * np.exp(-tau * r)


def calculate_implied_volatility(C, spot, fp, K, r, sig, tau):
    """
    sig is only use for initial guess
    """
    F = spot + fp

    def target(vol):
        return price_black(F, K, r, np.sqrt(vol ** 2 * tau)) - C

    return newton(target, sig * 2)


if __name__ == '__main__':
    spot = 1
    tau = 0.5
    fp = 0.03
    r = 0.05
    sig = 0.07
    ja = -0.04
    jv = 0.15 ** 2
    lmbda = 3

    mu = solve_mu(spot, fp, sig, tau, ja, jv, lmbda)
    res1 = price_with_black(spot, mu, 1, sig, tau, ja, jv, lmbda)
    res2 = price_with_characteristic(spot, mu, 1, sig, tau, ja, jv, lmbda)
    print('Using conditional expectation: ', res1)
    print('Using characteristic: ', res2)

    ori = []
    sig_up = []
    ja_up = []
    strikes = np.linspace(0.8, 1.3, 50)
    for K in strikes:
        # original implied vol
        call_price = price_with_black(spot, mu, K, sig, tau, ja, jv, lmbda)
        ori.append(calculate_implied_volatility(call_price, spot, fp, K, r, sig, tau))

        # 1.5 times sigma
        call_price = price_with_black(spot, mu, K, 1.5 * sig, tau, ja, jv, lmbda)
        sig_up.append(calculate_implied_volatility(call_price, spot, fp, K, r, 1.5 * sig, tau))

        # 1.1 times ja
        call_price = price_with_black(spot, mu, K, sig, tau, ja, 1.1 * jv, lmbda)
        ja_up.append(calculate_implied_volatility(call_price, spot, fp, K, r, 1.5 * sig, tau))

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(strikes, ori, label='Original')
    ax.legend()
    ax.plot(strikes, sig_up, label='1.5 Sigma')
    ax.legend()
    ax.plot(strikes, ja_up, label='1.1 Ja')
    ax.legend()
    plt.show()