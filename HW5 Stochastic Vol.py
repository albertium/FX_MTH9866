from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import newton
import numpy as np
import matplotlib.pyplot as plt


def solve_mu(spot, fp, sig, tau, ja, jv, lmbda):
    """
    given forward price, calculate risk neutral drift mu
    """
    F = spot + fp
    return np.log(F / spot) / tau - 0.5 * sig ** 2 - lmbda * (np.exp(ja + 0.5 * jv) - 1)


def price_black(F, K, r, sig_total):
    """
    Given forward F and total variance (equivalent to sig^2 * T), calculate call option price using Black model
    """
    d1 = np.log(F / K) / sig_total + 0.5 * sig_total
    d2 = d1 - sig_total
    price = F * norm.cdf(d1) - K * norm.cdf(d2)
    return price * np.exp(-r * tau)


def price_conditional_black(spot, mu, K, sig, tau, ja, jv, n):
    """
    calculate call option price given jump times
    """
    F = spot * np.exp(mu * tau + 0.5 * sig ** 2 * tau + n * (ja + 0.5 * jv))
    sig_total = np.sqrt(sig ** 2 * tau + n * jv)
    return price_black(F, K, r, sig_total)


def price_with_black(spot, mu, K, sig, tau, ja, jv, lmbda):
    """
    approximate the call option under Merton model using conditional Black prices and Poisson distribution
    """
    jumps = np.arange(31)
    jumps[0] = 1  # for calculate factorials
    factorials = np.cumprod(jumps)
    jumps[0] = 0  # restore the correct jump times
    probs = (lmbda * tau) ** jumps / factorials * np.exp(-lmbda * tau)  # Poisson PDF
    return np.dot(probs, [price_conditional_black(spot, mu, K, sig, tau, ja, jv, n) for n in jumps])


def price_with_characteristic(spot, mu, K, sig, tau, ja, jv, lmbda):
    """
    Price Merton model using characteristic function and numerical integration
    """
    sig2 = sig ** 2
    x = np.log(spot)

    # Characteristic function of Merton model
    def phi(theta):
        theta2 = theta ** 2
        at = 1j * theta * mu * tau - 0.5 * theta2 * sig2 * tau \
             + lmbda * tau * (np.exp(1j * theta * ja - theta2 * jv / 2) - 1)
        return np.exp(1j * theta * x + at)

    ratio = np.log(K / spot)

    def integrand(theta):
        return (phi(theta) * np.exp(-1j * theta * ratio) / (theta ** 2 + 1j * theta)).real

    # Numerical integration
    price = spot + fp - K / 2 - K / np.pi * quad(integrand, 0, np.inf)[0]
    return price * np.exp(-tau * r)


def calculate_implied_volatility(C, spot, fp, K, r, sig, tau):
    """
    Solve for single implied vol
    sig is only use for initial guess
    """
    F = spot + fp

    def target(vol):
        return price_black(F, K, r, np.sqrt(vol ** 2 * tau)) - C

    return newton(target, sig * 2)


def calculate_volatility_smile(strikes, spot, fp, sig, tau, ja, jv, lmbda):
    """
    Solve for the whole implied vol smile
    """
    mu = solve_mu(spot, fp, sig, tau, ja, jv, lmbda)
    vols = []
    for K in strikes:
        call_price = price_with_characteristic(spot, mu, K, sig, tau, ja, jv, lmbda)
        vols.append(calculate_implied_volatility(call_price, spot, fp, K, r, sig, tau))

    return vols


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

    # Implied vol smile with differnt parameter shock
    strikes = np.linspace(0.8, 1.3, 50)
    base = calculate_volatility_smile(strikes, spot, fp, sig, tau, ja, jv, lmbda)
    ja_up = calculate_volatility_smile(strikes, spot, fp, sig, tau, 0.1, jv, lmbda)
    jv_up = calculate_volatility_smile(strikes, spot, fp, sig, tau, ja, 0.16 ** 2, lmbda)
    lm_up = calculate_volatility_smile(strikes, spot, fp, sig, tau, ja, jv, 4)
    sig_up = calculate_volatility_smile(strikes, spot, fp, 0.14, tau, ja, jv, lmbda)

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(strikes, base, label='Base')
    ax.legend()
    ax.plot(strikes, ja_up, label='Jump Mean = 0.04')
    ax.legend()
    ax.plot(strikes, sig_up, label='Sigma = 0.14')
    ax.legend()
    ax.plot(strikes, jv_up, label='Jump Var = 0.16^2')
    ax.legend()
    ax.plot(strikes, lm_up, label='Lambda = 4')
    ax.legend()
    plt.show()