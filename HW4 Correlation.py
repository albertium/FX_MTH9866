
import pandas as pd
import numpy as np


def get_implied_corr(data, cross, base1, base2, tenor, sign, start_date=None, end_date=None):
    start_date = data.index.min() if start_date is None else start_date
    end_date = data.index.max() if end_date is None else end_date

    # only extract data for the specified tenor and date range
    data = data.loc[start_date: end_date, [col + ' ' + tenor for col in [cross, base1, base2]]].rename(lambda x: x[:-3], axis=1)
    vx, v1, v2 = data[cross], data[base1], data[base2]

    # calculate implied correlation
    data['rho'] = sign * (vx.pow(2) - v1.pow(2) - v2.pow(2)) / 2 / v1 / v2

    # calculate hedging implied volatility using previous correlation
    data['implied_cross_vol'] = np.sqrt(v1.pow(2) + v2.pow(2) + 2 * sign * v1 * v2 * data.rho.shift())

    # calculate unhedged and hedged implied volatility change
    data['unhedged'] = vx - vx.shift()   # change compared to previous
    data['hedged'] = vx - data.implied_cross_vol  # change against implied cross volatility
    return data


if __name__ == '__main__':
    data = pd.read_excel('fx_vol_data.xlsx', 'Sheet1', parse_dates=['Date']).set_index('Date')
    for tenor in ['1w', '1m', '6m', '1y']:
        res = get_implied_corr(data, 'AUDJPY', 'AUDUSD', 'USDJPY', tenor, -1)
        print(f'\nTenor {tenor}')
        print(res[['unhedged', 'hedged']].describe())
