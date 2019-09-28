
import pandas as pd
import numpy as np


def get_implied_corr(data, cross, base1, base2, tenor, sign, start_date=None, end_date=None):
    start_date = data.index.min() if start_date is None else start_date
    end_date = data.index.max() if end_date is None else end_date
    data = data.loc[start_date: end_date, [col + ' ' + tenor for col in [cross, base1, base2]]].rename(lambda x: x[:-3], axis=1)
    vx, v1, v2 = data[cross], data[base1], data[base2]
    data['rho'] = sign * (vx.pow(2) - v1.pow(2) - v2.pow(2)) / 2 / v1 / v2
    data['implied_cross_vol'] = np.sqrt(v1.pow(2) + v2.pow(2) + 2 * sign * v1 * v2 * data.rho.shift())
    data['change'] = vx - vx.shift()
    data['diff'] = data.implied_cross_vol - vx
    return data


if __name__ == '__main__':
    data = pd.read_excel('fx_vol_data.xlsx', 'Sheet1', parse_dates=['Date']).set_index('Date')
    for tenor in ['1w', '1m', '6m', '1y']:
        res = get_implied_corr(data, 'AUDJPY', 'AUDUSD', 'USDJPY', '1w', -1)
        print(f'\nTenor {tenor}')
        print(res[['change', 'diff']].describe())
