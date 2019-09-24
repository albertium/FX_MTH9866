
import pandas as pd
import numpy as np


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


def process_compustat(filename, year_end=12):
    fund = pd.read_csv(filename, compression='gzip', parse_dates=['datadate'], infer_datetime_format='%Y%m%d',
                       dtype={'LPERMNO': str}).rename({'LPERMNO': 'PERMNO', 'datadate': 'date'}, axis=1)

    total = fund.shape[0]
    print(f'Total rows: {total}')
    print(f'{fund.date.min()} to {fund.date.max()}')

    required_cols = ['act', 'at', 'che', 'dlc', 'dltt', 'ivao', 'ivst', 'lct', 'lt', 'oiadp', 'pstk']
    critical_cols = ['at', 'che', 'act', 'lct', 'lt', 'oiadp']
    fund = fund[['PERMNO', 'date', 'fyear'] + required_cols].dropna(how='any', subset=critical_cols)
    fund = fund.fillna({'dltt': 0, 'dlc': 0, 'pstk': 0, 'ivst': 0, 'ivao': 0})
    print(f'Drop NAs: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    dup = fund[['PERMNO', 'fyear']].duplicated().sum()
    print(f'Duplicated PERMNO fyear: {dup}')

    fund.drop_duplicates(subset=['PERMNO', 'fyear'], keep='last', inplace=True)
    print(f'Drop duplicates: {fund.shape[0]} ({fund.shape[0] / total:.2%})')
    dup = fund[['PERMNO', 'fyear']].duplicated().sum()
    print(f'Duplicated PERMNO fyear: {dup}')

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

    fund['fyear1'] = fund.fyear.shift()
    fund = fund.merge(fund, left_on=['PERMNO', 'fyear1'], right_on=['PERMNO', 'fyear'], how='left',
                      suffixes=('', '1')).dropna()
    print(f'Merged with previous fyear: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # delta metrics
    fund['avg_at'] = (fund['at'] + fund['at1']) / 2
    fund['roa'] = fund.oiadp / fund.avg_at
    fund['dwc'] = (fund.wc - fund.wc1) / fund.avg_at
    fund['dnco'] = (fund.nco - fund.nco1) / fund.avg_at
    fund['dfin'] = (fund.fin - fund.fin1) / fund.avg_at
    fund['tacc'] = fund.dwc + fund.dnco + fund.dfin
    fund.drop([col for col in fund.columns if '1' in col], axis=1, inplace=True)
    fund['year'] = fund.date.dt.year - (fund.date.dt.month <= (year_end % 12))
    fund = fund.groupby(['PERMNO', 'year'], as_index=False).last()

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


def backtest(panel, strategy):
    start_year = panel.year.min()
    end_year = panel.year.max()

    positions = {}
    equity = []
    for year in range(start_year, end_year + 1):
        curr_data = panel[panel.year == year]

        # calculate return
        portfolio_return = [curr_data.loc[no].RET * weight for no, weight in positions.items() if no in curr_data.index]
        equity.append(np.sum(portfolio_return))

        # rebalance
        positions = {}