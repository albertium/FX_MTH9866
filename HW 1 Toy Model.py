
import numpy as np
import time


def simulate(fx_spot, fx_incs, trades, client_spread, dealer_spread, hedger, verbose=False):
    N = fx_incs.shape[1]
    pnl = np.zeros(N)
    position = np.zeros(N)
    spots = np.ones(N) * fx_spot
    spread_gain = np.zeros(N)
    cost = np.zeros(N)
    mtm = np.zeros(N)
    for inc, trade in zip(fx_incs, trades):  # get spot incremental and clients
        mtm += position * inc  # pnl change due to risk positions
        position += trade  # update position with client trades
        hedge = hedger(position)  # hedging strategy
        spread_gain += np.abs(trade) * spots * client_spread * 0.5
        cost += np.abs(hedge) * spots * dealer_spread * 0.5

        # update quantities
        spots += inc  # update fx spot
        position += hedge  # update position after hedging

    pnl += spread_gain - cost + mtm

    if verbose:
        print(f'Sharp ratio:            {np.mean(pnl) / np.std(pnl):.3f}')
        print(f'Average spread gain:    {spread_gain.mean():.6f}')
        print(f'Average hedging cost:   {cost.mean():.6f}')
        print(f'Average MTM PnL:        {mtm.mean()}\n')
    return np.mean(pnl) / np.std(pnl)


def run_simulation(verbose=False):
    # parameters
    spot = 1
    vol = 0.1
    num_sec = 260 * 24 * 3600
    lam = 1 * num_sec
    dt = 0.1 / lam
    steps = 500
    N = 10000
    client_spread = 0.0001
    dealer_spread = 0.0002
    delta_limit = 3

    # simulation
    fx_inc = np.random.normal(0, vol * np.sqrt(dt), (steps, N))  # fx spot incremental
    # client trades with arrival rate of lambda * dt. 0.5 chance being long or short
    trades = np.random.binomial(1, 1 - np.exp(-lam * dt), (steps, N)) * (2 * (np.random.rand(steps, N) > 0.5) - 1)

    # sanity check
    if verbose:
        print(f'Terminal spot vol: {np.sum(fx_inc, axis=0).std():.4%} ({vol * np.sqrt(dt * steps):.4%})')
        print(f'Average trades: {np.sum(np.abs(trades), axis=0).mean():.2f} ({steps * lam * dt:.2f})')

    full_hedger = lambda pos: -pos * (np.abs(pos) >= delta_limit)  # hedge the whole position
    partial_hedger = lambda pos: -np.sign(pos) * np.maximum(np.abs(pos) - delta_limit, 0)  # hedge down to delta_limit

    if verbose:
        print('Full hedge')
    full_sharpe = simulate(spot, fx_inc, trades, client_spread, dealer_spread, full_hedger, verbose)

    if verbose:
        print('Partial hedge')
    partial_sharpe = simulate(spot, fx_inc, trades, client_spread, dealer_spread, partial_hedger, verbose)
    return full_sharpe, partial_sharpe


if __name__ == '__main__':
    results = []
    start = time.time()
    run_simulation(True)  # run once with printout

    # to estimate stability based on Sharpe ratio
    for i in range(100):
        print(i)
        results.append(run_simulation(False))
    print(f'Time: {time.time() - start:.2f}s')
    results = np.array(results)
    print(f'means: {np.mean(results, axis=0)}')
    print(f'stddevs: {np.std(results, axis=0)}')
