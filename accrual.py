
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd


def process_crsp(filename, year_end=12):
    raw = pd.read_csv(filename, compression='gzip', parse_dates=['date'], infer_datetime_format='%Y%m%d',
                      dtype={'SICCD': str, 'PERMNO': str, 'PERMCO': str, 'PRC': str, 'RET': str, 'CFACPR': str})

    columns_kept = ['date', 'PERMNO', 'SICCD', 'TICKER', 'DLSTCD', 'RET']
    tota_rows = raw.shape[0]
    print(f'Total rows: {tota_rows}')
    print(f'{raw.date.min()} to {raw.date.max()}')

    data = raw[columns_kept].set_index(['date', 'PERMNO']).dropna(how='all').reset_index()
    print(f'Remove all NAs: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    # C is usually the beginning of the data (missing previous data)
    # B is unknown missing
    # keep both to maximize data usage in merging
    data.RET = data.RET.replace({'C': 0, 'B': 0}).astype('float')
    data = data[data.SICCD != 'Z']
    print(f'Remove Zs: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    data.DLSTCD.fillna(0, inplace=True)  # for easier processing, especially in dropna step
    data.SICCD = data.SICCD.astype('int')

    data = data[(data.SICCD < 6000) | (data.SICCD >= 7000)]  # remove financial
    print(f'Remove financials: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    # remove rows after delist code is on
    # TODO: should we use different recovery rate for different code?
    data = data.groupby('PERMNO', as_index=False).apply(
        lambda x: x.iloc[:x.reset_index().DLSTCD.ne(0).idxmax()]).reset_index(drop=True)
    print(f'Remove delisted rows: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    data = data.drop_duplicates()
    print(f'Remove duplicated rows: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    dup = data[['PERMNO', 'date']].duplicated().sum()
    print(f'Duplicated for key [PERMNO, date]: {dup}')

    # collapse to annual data
    data['year'] = data.date.dt.year - (data.date.dt.month <= (year_end % 12))
    data = data.groupby(['PERMNO', 'year']).agg({'RET': lambda x: (1 + x).prod() - 1})
    print(f'PERMNO year: {data.shape[0]}')
    return data


def process_crsp2(filename):
    data = pd.read_feather(filename)
    data['share'] = data['shrout'] * data['cfacshr'] * 1e3
    data['price'] = data['prc'].abs() / data['cfacpr']
    data['cap'] = data.price * data.share / 1e6

    columns_kept = ['date', 'permno', 'siccd', 'ret', 'cap']
    tota_rows = data.shape[0]
    print(f'Total rows: {tota_rows}')
    print(f'{data.date.min()} to {data.date.max()}')

    print('Check missing')
    for col in columns_kept:
        count = data[col].isna().sum()
        print(f'    {col} missing: {count} ({count / tota_rows:.2%})')

    data = data[columns_kept].dropna()
    print(f'Remove NAs: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    data = data[(data.siccd < 6000) | (data.siccd > 6999)]  # remove financial
    print(f'Remove financials: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    dup = data[['permno', 'date']].duplicated().sum()
    print(f'Duplicated for key [permno, date]: {dup}')

    data.permno = data.permno.astype(int)
    data['date'] = pd.to_datetime(data['date'])
    data['join_date'] = data.date + MonthEnd(0)
    # data['next_join_date'] = data.join_date + MonthEnd(1)

    # tmp = data[['permno', 'join_date', 'ret']].rename({'ret': 'next_ret', 'join_date': 'next_join_date'}, axis=1)
    # data = pd.merge(data, tmp, on=['permno', 'next_join_date'])
    data = data.dropna()
    # print(f'After next return: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    # return data[['permno', 'join_date', 'ret', 'next_ret', 'cap']]
    return data[['permno', 'join_date', 'ret', 'cap']]


def process_crsp3(filename,year_end=6):
    """
    For June Split
    """
    data = pd.read_feather(filename)
    data['share'] = data['shrout'] * data['cfacshr'] * 1e3
    data['price'] = data['prc'].abs() / data['cfacpr']
    data['cap'] = data.price * data.share / 1e6

    columns_kept = ['date', 'permno', 'siccd', 'ret', 'cap']
    tota_rows = data.shape[0]
    print(f'Total rows: {tota_rows}')
    print(f'{data.date.min()} to {data.date.max()}')

    print('Check missing')
    for col in columns_kept:
        count = data[col].isna().sum()
        print(f'    {col} missing: {count} ({count / tota_rows:.2%})')

    data = data[columns_kept].dropna()
    print(f'Remove NAs: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    data = data[(data.siccd < 6000) | (data.siccd > 6999)]  # remove financial
    print(f'Remove financials: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    dup = data[['permno', 'date']].duplicated().sum()
    print(f'Duplicated for key [permno, date]: {dup}')

    data.permno = data.permno.astype(int)
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data.date.dt.year - (data.date.dt.month <= (year_end % 12))
    return data[['permno', 'year', 'date', 'ret', 'cap']].sort_values(['permno', 'date'])


def process_compustat(filename, year_end=12):
    # fund = pd.read_csv(filename, compression='gzip', parse_dates=['datadate'], infer_datetime_format='%Y%m%d',
    #                    dtype={'LPERMNO': 'int'}).rename({'LPERMNO': 'permno', 'datadate': 'date'}, axis=1)

    fund = pd.read_feather(filename)
    total = fund.shape[0]
    print(f'Total rows: {total}')
    print(f'{fund.date.min()} to {fund.date.max()}')

    required_cols = ['act', 'at', 'che', 'dlc', 'dltt', 'ivao', 'ivst', 'lct', 'lt', 'oiadp', 'pstk', 'sale']
    critical_cols = ['at', 'che', 'act', 'lct', 'lt', 'sale']
    fund = fund[['permno', 'date'] + required_cols].dropna(how='any', subset=critical_cols)
    fund = fund.fillna({'dltt': 0, 'dlc': 0, 'pstk': 0, 'ivst': 0, 'ivao': 0})
    print(f'Drop NAs: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    dup = fund[['permno', 'date']].duplicated().sum()
    print(f'Duplicated permno date: {dup}')

    zero = (fund["at"] == 0).sum()
    print(f'Zero total asset: {zero}')
    fund = fund[fund['at'] != 0]
    print(f'Drop zero AT: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # calculate metrics
    fund['coa'] = fund.act - fund.che
    fund['col'] = fund.lct - fund.dlc
    fund['wc'] = fund.coa - fund.col
    fund['ncoa'] = fund['at'] - fund.act - fund.ivao
    fund['ncol'] = fund['lt'] - fund.lct - fund.dltt
    fund['nco'] = fund.ncoa - fund.ncol
    fund['fina'] = fund.ivst + fund.ivao
    fund['finl'] = fund.dltt + fund.dlc + fund.pstk
    fund['fin'] = fund.fina - fund.finl

    fund = fund.sort_values(['permno', 'date'])
    cols = ['at', 'wc', 'nco', 'fin', 'sale']  # sale is not used but is required in data cleaning
    prev_cols = ['prev_' + x for x in cols]
    fund[prev_cols] = fund.groupby('permno', as_index=False)[cols].shift()
    fund.dropna(inplace=True)
    print(f'Merged with previous total asset: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # delta metrics
    fund['avg_at'] = (fund['at'] + fund['prev_at']) / 2
    fund['roa'] = fund.oiadp / fund.avg_at
    fund['dwc'] = (fund.wc - fund.prev_wc) / fund.avg_at
    fund['dnco'] = (fund.nco - fund.prev_nco) / fund.avg_at
    fund['dnoa'] = fund['dwc'] + fund['dnco']
    fund['dfin'] = (fund.fin - fund.prev_fin) / fund.avg_at
    fund['tacc'] = fund.dwc + fund.dnco + fund.dfin
    fund.drop([col for col in fund.columns if 'prev_' in col], axis=1, inplace=True)
    fund['year'] = fund.date.dt.year - (fund.date.dt.month <= (year_end % 12))
    fund['join_date'] = fund.date + MonthEnd(0)

    print(f'Final rows: {fund.shape[0]} ({fund.shape[0] / total:.2%})')
    return fund


def merge_data(crsp, compustat):
    valid_no = compustat.PERMNO.unique()
    total = crsp.shape[0]
    print(f'Total rows: {total}')

    merged = crsp[crsp.PERMNO.isin(valid_no)]
    print(f'Valid PERMNO: {merged.shape[0]} ({merged.shape[0] / total:.2%})')

    join_keys = ['PERMNO', 'year']
    merged = merged.join(compustat.set_index(join_keys), on=join_keys, how='left')
    print(f'Merged: {merged.shape[0]} ({merged.shape[0] / total:.2%})')

    filled = merged.groupby('PERMNO', as_index=False).fillna(method='ffill')
    print(f'Filled: {filled.shape[0]} ({filled.shape[0] / total:.2%})')

    filled = merged.dropna()
    print(f'Drop NAs: {filled.shape[0]} ({filled.shape[0] / total:.2%})')
    return filled


def backtest(panel):
    start_year = panel.year.min()
    end_year = panel.year.max()

    positions = {}
    equity = []
    for year in range(start_year, end_year + 1):
        curr_data = panel[panel.year == year].set_index('PERMNO')

        # calculate return
        portfolio_return = [curr_data.loc[no].RET * weight for no, weight in positions.items() if no in curr_data.index]
        equity.append(np.sum(portfolio_return))

        # rebalance
        bucket = pd.qcut(curr_data.tacc, 10, range(10))
        longs = curr_data[bucket == 0].index.values
        shorts = curr_data[bucket == 9].index.values
        positions = {**dict.fromkeys(longs, 0.5 / len(longs)), **dict.fromkeys(shorts, -0.5 / len(shorts))}

    return equity