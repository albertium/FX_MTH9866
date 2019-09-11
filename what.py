from math import exp, sqrt
import scipy


def run_simulation_fast(vol, lam, sprd_client, sprd_dealer, delta_lim, hedge_style, dt, nsteps, nruns, seed):
    '''Runs a Monte Carlo simulation and returns statics on PNL, client trades, and hedge trades.
    "_fast" because it uses vectorized operations.

    vol:         lognormal volatility of the spot process
    lam:         Poisson process frequency
    sprd_client: fractional bid/ask spread for client trades. eg 1e-4 means 1bp.
    sprd_dealer: fractional bid/ask spread for inter-dealer hedge trades. eg 1e-4 means 1bp.
    delta_lim:   the delta limit at or beyond which the machine will hedge in the inter-dealer market
    hedge_style: 'Zero' or 'Edge', defining the hedging style. 'Zero' means hedge to zero position,
                 'Edge' means hedge to the nearer delta limit.
    dt:          length of a time step
    nsteps:      number of time steps for each run of the simulation
    nruns:       number of Monte Carlo runs
    seed:        RNG seed
    '''

    scipy.random.seed(seed)

    trade_prob = 1 - exp(-lam * dt)
    sqrtdt = sqrt(dt)

    spots = scipy.zeros(nruns) + 1  # initial spot == 1
    posns = scipy.zeros(nruns)
    trades = scipy.zeros(nruns)
    hedges = scipy.zeros(nruns)
    pnls = scipy.zeros(nruns)

    for step in range(nsteps):
        dzs = scipy.random.normal(0, sqrtdt, nruns)
        qs = scipy.random.uniform(0, 1, nruns)
        ps = scipy.random.binomial(1, 0.5, nruns) * 2 - 1  # +1 or -1 - trade quantities if a trade happens

        # check if there are client trades for each path

        indics = scipy.less(qs, trade_prob)
        posns += indics * ps
        trades += scipy.ones(nruns) * indics
        pnls += scipy.ones(nruns) * indics * sprd_client * spots / 2.

        # check if there are hedges to do for each path

        if hedge_style == 'Zero':
            indics = scipy.logical_or(scipy.less_equal(posns, -delta_lim), scipy.greater_equal(posns, delta_lim))
            pnls -= scipy.absolute(posns) * indics * sprd_dealer * spots / 2.
            posns -= posns * indics
            hedges += scipy.ones(nruns) * indics
        elif hedge_style == 'Edge':
            # first deal with cases where pos>delta_lim

            indics = scipy.greater(posns, delta_lim)
            pnls -= (posns - delta_lim) * indics * sprd_dealer * spots / 2.
            posns = posns * scipy.logical_not(indics) + scipy.ones(nruns) * indics * delta_lim
            hedges += scipy.ones(nruns) * indics

            # then the cases where pos<-delta_lim

            indics = scipy.less(posns, -delta_lim)
            pnls -= (-delta_lim - posns) * indics * sprd_dealer * spots / 2.
            posns = posns * scipy.logical_not(indics) + scipy.ones(nruns) * indics * (-delta_lim)
            hedges += scipy.ones(nruns) * indics
        else:
            raise ValueError('hedge_style must be "Edge" or "Zero"')

        # advance the spots and calculate period PNL

        dspots = vol * spots * dzs
        pnls += posns * dspots
        spots += dspots

    return {'PNL': (pnls.mean(), pnls.std()), 'Trades': (trades.mean(), trades.std()),
            'Hedges': (hedges.mean(), hedges.std())}


def simulation():
    '''Little test function that demonstrates how to run this'''

    vol = 0.1 * sqrt(1 / 260.)  # 10% annualized vol, converted to per-day vol using 260 days/year
    lam = 1 * 60 * 60 * 24  # Poisson frequency for arrival of client trades: 1 per second, converted into per-day frequency to be consistent with vol

    sprd_client = 1e-4  # fractional full bid/ask spread for client trades
    sprd_dealer = 2e-4  # fractional full bid/ask spread for inter-dealer hedge trades

    hedge_style = 'Edge'  # 'Zero' means "hedge to zero position", or 'Edge' means "hedge to delta limit"

    delta_lim = 3.  # algorithm hedges when net delta reaches this limit

    dt = 0.1 / lam  # time step in simulation. Only zero or one client trades can arrive in a given interval.
    nsteps = 500  # number of time steps in the simulation
    nruns = 50000  # number of Monte Carlo runs
    seed = 123  # Monte Carlo seed

    res = run_simulation_fast(vol, lam, sprd_client, sprd_dealer, delta_lim, hedge_style, dt, nsteps, nruns, seed)
    print('PNL Sharpe ratio             =', res['PNL'][0] / res['PNL'][1])
    print('PNL mean                     =', res['PNL'][0])
    print('PNL std dev                  =', res['PNL'][1])
    print('Mean number of client trades =', res['Trades'][0])
    print('SD number of client trades   =', res['Trades'][1])
    print('Mean number of hedge trades  =', res['Hedges'][0])
    print('SD number of hedge trades    =', res['Hedges'][1])


if __name__ == "__main__":
    import time

    t1 = time.time()
    simulation()
    t2 = time.time()
    print('Elapsed time =', t2 - t1, 'seconds')
