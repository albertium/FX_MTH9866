
import numpy as np
import itertools


def simulate(N, dt, T, T1, T2, spot, Q, sig1, sig2, beta1, beta2, rho):
    """
    generate forward returns for T, T1 and T2 per 2 factor model
    """
    motions = np.random.normal(0, np.sqrt(dt), (2, N))
    motions[1] = rho * motions[0] + np.sqrt(1 - rho * rho) * motions[1]  # introduce correlation
    # coefficients of the Brownian motions for each tenor
    coefs = np.array([[sig1 * np.exp(-beta1 * t), sig2 * np.exp(-beta2 * t)] for t in [T, T1, T2]])
    dQs = np.matmul(coefs, motions)
    Ts = np.array([T, T1, T2]).reshape(-1, 1)
    return spot * np.exp(-(Q + dQs) * Ts) - np.exp(-Q * Ts)  # forward return


def get_no_hedge(T, T1, T2, Q, sig1, sig2, beta1, beta2, rho):
    return 0, 0  # no hedge


def get_triangle_hedge(T, T1, T2, Q, sig1, sig2, beta1, beta2, rho):
    if T < T1:
        ratio = 1  # flat for regions < T1
    elif T > T2:
        ratio = 0  # flat for regions > T2
    else:
        ratio = (T2 - T) / (T2 - T1)
    return ratio * T / T1 * np.exp(-Q * (T - T1)), (1 - ratio) * T / T2 * np.exp(-Q * (T - T2))


def get_factor_hedge(T, T1, T2, Q, sig1, sig2, beta1, beta2, rho):
    """
    return hedge amount using factor model
    """
    # partial derivatives of forward T, T1, T2 with respective to interest rate Q, Q1, Q2
    V_Q = np.array([[t * np.exp(-Q * t) for t in [T, T1, T2]]])
    # partial derivatives of interest rate Q, Q1, Q2 with respective to Brownian motion z1, z2
    Q_z = np.array([r1 * sig1 * np.exp(-beta1 * t) + r2 * sig2 * np.exp(-beta2 * t)
                    for t, (r1, r2) in itertools.product([T, T1, T2], [[1, rho], [rho, 1]])]).reshape(-1, 2).T
    A = V_Q[:, 1:] * Q_z[:, 1:]  # partial derivative of forward T1, T2 with respective to z1, z2
    b = V_Q[:, 0] * Q_z[:, 0]  # partial derivative of forward T with respective to z1, z2
    # partial derivatives of whole portfolio (with weights 1, -N1, -N2) with respective to z1, z2 should be 0
    # using this relationship, solve for N1, N2
    hedges = np.linalg.solve(A, b)
    return hedges[0], hedges[1]


def get_pca_hedge(T, T1, T2, Q, sig1, sig2, beta1, beta2, rho):
    """
    just for fun. return hedges using PCA method
    """
    # simulate forward return as "historical data" for covariance calculation
    Ts = [0.1, 0.25, 0.5, 0.75, 1, 2]
    motions = np.random.normal(0, np.sqrt(dt), (2, N))
    motions[1] = rho * motions[0] + np.sqrt(1 - rho * rho) * motions[1]  # introduce correlation
    # coefficients of the Brownian motions for each tenor
    coefs = np.array([[sig1 * np.exp(-beta1 * t), sig2 * np.exp(-beta2 * t)] for t in Ts])
    dQs = np.matmul(coefs, motions)

    # follow the same procedure as in factor hedging. But use factor loadings instead for Q's sensitivity to z's
    cov = np.cov(dQs)
    _, loadings = np.linalg.eig(cov)
    V_Q = np.array([[t * np.exp(-Q * t) for t in [T, T1, T2]]])
    Q_z = loadings[[Ts.index(t) for t in [T, T1, T2]], :2].T  # find the factor loadings for corresponding tenors
    A = V_Q[:, 1:] * Q_z[:, 1:]
    b = V_Q[:, 0] * Q_z[:, 0]
    hedges = np.linalg.solve(A, b)
    return hedges[0], hedges[1]


def hedge(hedger, N, dt, T: list, T1, T2, spot, Q, sig1, sig2, beta1, beta2, rho):
    """
    get forward returns for T, T1, T2 and apply hedging per returned from hedger
    """
    hedged_std = []
    for t in T:
        forward_return = simulate(N, dt, t, T1, T2, spot, Q, sig1, sig2, beta1, beta2, rho)
        N1, N2 = hedger(t, T1, T2, Q, sig1, sig2, beta1, beta2, rho)
        weights = np.array([1, -N1, -N2])  # T forward hedged by shorting N1 and N2 T1 and T2 forwards
        hedged_return = np.matmul(weights, forward_return)
        hedged_std.append([np.std(hedged_return)])  # hedging error
    return np.array(hedged_std)


if __name__ == '__main__':
    N = 100000
    dt = 0.001
    T1 = 0.25
    T2 = 1
    spot = 1
    Q = 0.03
    sig1 = 0.01
    sig2 = 0.008
    beta1 = 0.5
    beta2 = 0.1
    rho = -0.4

    # run hedging with the 3 hedgers
    Ts = [0.1, 0.25, 0.5, 0.75, 1, 2]  # forward tenors
    results = []
    # include PCA just for fun
    for hedger in [get_no_hedge, get_triangle_hedge, get_factor_hedge, get_pca_hedge]:
        results.append(hedge(hedger, N, dt, Ts, T1, T2, spot, Q, sig1, sig2, beta1, beta2, rho))

    results = np.hstack(results)
    print(f'{"Tenor":<8}|{"No Hedge":<12}{"Triangle":<12}{"Triangle %":<12}{"Factor":<12}{"Factor %":<12}'
          f'{"PCA":<12}{"PCA %":<12}')
    print('--------|-------------------------------------------------------------------------------')
    for tenor, data in zip(Ts, results):
        print(f'{tenor:<8.2f}|{data[0] * 1e4:<12.3f}'
              f'{data[1] * 1e4:<12.3f}{1 - data[1] / data[0]:<12.1%}'  # triangle stats
              f'{data[2] * 1e4:<12.3f}{1 - data[2] / data[0]:<12.1%}'  # factor model stats
              f'{data[3] * 1e4:<12.3f}{1 - data[3] / data[0]:<12.1%}')  # PCA model stats
