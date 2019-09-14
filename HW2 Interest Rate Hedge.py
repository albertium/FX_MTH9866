
import numpy as np
import itertools


def simulate(N, dt, T, T1, T2, spot, Q, sig1, sig2, beta1, beta2, rho):
    motions = np.random.normal(0, np.sqrt(dt), (2, N))
    motions[1] = rho * motions[0] + np.sqrt(1 - rho * rho) * motions[1]
    coefs = np.array([[sig1 * np.exp(-beta1 * t), sig2 * np.exp(-beta2 * t)] for t in [T, T1, T2]])
    dQs = np.matmul(coefs, motions)
    Ts = np.array([T, T1, T2]).reshape(-1, 1)
    return spot * np.exp(-(Q + dQs) * Ts) - np.exp(-Q * Ts)  # forward contract change


def get_no_hedge(T, T1, T2, Q, sig1, sig2, beta1, beta2, rho):
    return 0, 0


def get_triangle_hedge(T, T1, T2, Q, sig1, sig2, beta1, beta2, rho):
    if T < T1:
        ratio = 1
    elif T > T2:
        ratio = 0
    else:
        ratio = (T2 - T) / (T2 - T1)
    return ratio * T / T1 * np.exp(-Q * (T - T1)), (1 - ratio) * T / T2 * np.exp(-Q * (T - T2))


def get_factor_hedge(T, T1, T2, Q, sig1, sig2, beta1, beta2, rho):
    V_Q = np.array([[t * np.exp(-Q * t) for t in [T, T1, T2]]])
    Q_z = np.array([r1 * sig1 * np.exp(-beta1 * t) + r2 * sig2 * np.exp(-beta2 * t)
                    for t, (r1, r2) in itertools.product([T, T1, T2], [[1, rho], [rho, 1]])]).reshape(-1, 2).T
    A = V_Q[:, 1:] * Q_z[:, 1:]
    b = V_Q[:, 0] * Q_z[:, 0]
    hedges = np.linalg.solve(A, b)
    return hedges[0], hedges[1]


def hedge(hedger, N, dt, T: list, T1, T2, spot, Q, sig1, sig2, beta1, beta2, rho):
    hedged_std = []
    for t in T:
        forward_return = simulate(N, dt, t, T1, T2, spot, Q, sig1, sig2, beta1, beta2, rho)
        N1, N2 = hedger(t, T1, T2, Q, sig1, sig2, beta1, beta2, rho)
        weights = np.array([1, -N1, -N2])
        hedged_return = np.matmul(weights, forward_return)
        hedged_std.append([np.std(hedged_return)])
    print()
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

    # sanity check
    # fwd_ret = simulate(N, dt, 0.1, 0.25, 1, spot, Q, sig1, sig2, beta1, beta2, rho)
    # print(np.std(fwd_ret, axis=1))  # standard deviation of simulated forward return
    # Ts = np.array([0.1, 0.25, 1])
    # coef1 = sig1 * np.exp(-beta1 * Ts)
    # coef2 = sig2 * np.exp(-beta2 * Ts)
    # # theoretical standard deviation of simulated forward return
    # print(np.exp(-Q * Ts) * np.sqrt(coef1 ** 2 + coef2 ** 2 + 2 * rho * coef1 * coef2) * np.sqrt(dt) * Ts)

    # run hedging
    Ts = [0.1, 0.25, 0.5, 0.75, 1, 2]
    results = []
    for hedger in [get_no_hedge, get_triangle_hedge, get_factor_hedge]:
        results.append(hedge(hedger, N, dt, Ts, T1, T2, spot, Q, sig1, sig2, beta1, beta2, rho))

    results = np.hstack(results)
    print(f'{"Tenor":<8}{"No Hedge":<12}{"Triangle":<12}{"Triangle %":<12}{"Factor":<12}{"Factor %":<12}')
    for tenor, data in zip(Ts, results):
        print(f'{tenor:<8.2f}{data[0] * 1e4:<12.3f}'
              f'{data[1] * 1e4:<12.3f}{1 - data[1] / data[0]:<12.1%}'  # triangle stats
              f'{data[2] * 1e4:<12.3f}{1 - data[2] / data[0]:<12.1%}')  # factor model stats
