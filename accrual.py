
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd, QuarterEnd, YearEnd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time


def df_range(df, start_date, end_date):
    return df[(df.date >= start_date) & (df.date <= end_date)]


def df_semi_minus(df1, df2, left, right=None):
    if right is None:
        right = left

    df2 = df2[right].copy()
    df2['_flag_'] = 1
    joined = pd.merge(df1, df2, left_on=left, right_on=right, how='left', suffixes=('', '_y'))
    joined = joined[joined['_flag_'].isna()]
    return joined.drop([col for col in joined.columns if col.endswith('_y')] + ['_flag_'], axis=1)


def join_crsp_and_funda(crsp, funda, offset=QuarterEnd(2)):
    crsp = crsp.copy()
    crsp['time_idx_p'] = crsp['time_idx'] - offset
    joined = pd.merge(crsp, funda, left_on=['permno', 'time_idx_p'], right_on=['permno', 'time_idx'],
                      how='left', suffixes=('', '_d')).drop('time_idx_d', axis=1)
    joined.sort_values(['permno', 'time_idx'], inplace=True)
    joined = joined.groupby('permno', as_index=False).fillna(method='ffill').dropna()
    print(f'CRSP recrods: {crsp.shape[0]}')
    print(f'Merged recrods: {joined.shape[0]} ({joined.shape[0] / crsp.shape[0]:.2%})')
    return joined


def neutralize_alphas_and_returns(df, group_by, ret_col, alphas):
    def normalize(x):
        normal = (x - x.mean()) / x.std()
        return np.maximum(np.minimum(normal, 3), -3)

    df = df.copy()
    # neutralize alphas
    df[[x + '_n' for x in alphas]] = df.groupby(group_by, as_index=False)[alphas].transform(normalize)

    # neutralize returns
    df = df.set_index(group_by + ['permno'])
    mean = df.groupby(level=list(range(len(group_by))))[ret_col].mean()
    df['adj_ret'] = df[ret_col] - mean
    return df.dropna().reset_index()


def calculate_IC(df, ret_col, time_col, alphas):
    skip = len(alphas) + 1
    ic = df.groupby(time_col)[alphas + [ret_col]].corr(method='spearman').iloc[(skip - 1)::skip]
    ic = ic.reset_index().drop(['level_1', ret_col], axis=1)
    return ic


def calculate_sorted_returns(df, signals, time_col='year', ret_col='ret', weight_col='cap', n_cuts=10):
    cut_idx = list(range(n_cuts))
    df['equal'] = 1
    for signal in signals:
        df[signal + '_r'] = df.groupby(time_col, as_index=False)[signal].transform(lambda x: pd.qcut(x, n_cuts, cut_idx))

    results = []
    df['scaled_ret'] = df[weight_col] * df[ret_col]
    for rank in [x + '_r' for x in signals]:
        port_ret = df.groupby([rank, time_col], as_index=False).agg({'scaled_ret': 'sum', weight_col: 'sum'})
        port_ret['wgt_ret'] = port_ret.scaled_ret / port_ret[weight_col]

        port_ret = port_ret.pivot(index=time_col, columns=rank, values='wgt_ret').reset_index()
        port_ret['hedged'] = (port_ret[0] - port_ret[9]) / 2  # make sure leverage is 1
        port_ret['avg'] = port_ret.hedged.rolling(4, min_periods=4).sum()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(y=port_ret.avg, x=port_ret[time_col], name=rank), secondary_y=False)
        fig.add_trace(go.Scatter(y=(1 + port_ret.hedged).cumprod(), x=port_ret[time_col], name='Equity'), secondary_y=True)
        fig.show()
        results.append(port_ret)

    return results


def calculate_excess_returns_by_buckets(df, signals, time_col='year', ret_col='ret', weight_col='cap', n_cuts=10,
                                        exempt=None):
    cut_idx = list(range(n_cuts))
    df['equal'] = 1

    # set up exempt list
    if exempt is None:
        exempt = {}
    else:
        exempt = {k: 1 for k in exempt}

    for signal in signals:
        if signal in exempt:
            df[signal + '_r'] = df[signal]
        else:
            df[signal + '_r'] = df.groupby(time_col, as_index=False)[signal].transform(lambda x: pd.qcut(x, n_cuts, cut_idx))

    results = []
    df['scaled_ret'] = df[weight_col] * df[ret_col]
    df['hit'] = df[ret_col] > 0  # for calculating hit rate
    for rank in [x + '_r' for x in signals]:
        port_ret = df.groupby([rank, time_col], as_index=False)\
            .agg({'scaled_ret': 'sum', weight_col: 'sum', 'hit': 'mean'})
        port_ret['wgt_ret'] = port_ret.scaled_ret / port_ret[weight_col]

        # organize result
        overall = port_ret.groupby(rank, as_index=False).agg({'wgt_ret': ['mean', 'std'], 'hit': ['mean', 'count']}, axis=1)
        overall.columns = ["_".join(x) if x[1] != '' else x[0] for x in overall.columns.ravel()]
        overall['t_stat'] = overall.wgt_ret_mean / overall.wgt_ret_std
        overall = overall.set_index(rank).drop('wgt_ret_std', axis=1)\
            .rename({'wgt_ret_mean': 'ret', 'hit_mean': 'hit', 'hit_count': 'count'}, axis=1)
        results.append(overall[['ret', 't_stat', 'hit', 'count']])

    return results


def get_funda_data(conn, items):
    comp = conn.raw_sql(f"""
        select gvkey, datadate as date, fyear, cusip, sich, seq, 
        {items}
        from comp.funda
        where indfmt='INDL' 
        and datafmt='STD'
        and popsrc='D'
        and consol='C'
        and datadate >= '01/01/1960'
    """)

    ccm = conn.raw_sql("""
        select gvkey, lpermno as permno, linkdt, linkenddt
        from crsp.ccmxpf_linktable
        where (linktype ='LU' or linktype='LC')
    """)

    print(f'comp records: {comp.shape[0]}')
    print(f'ccm records: {ccm.shape[0]}')

    comp['date'] = pd.to_datetime(comp.date)
    ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
    ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt']).fillna(pd.to_datetime('today'))  # use today if missing
    ccm1 = pd.merge(comp, ccm, how='left', on=['gvkey'])
    final = ccm1[(ccm1.date >= ccm1.linkdt) & (ccm1.date <= ccm1.linkenddt)]
    print(f'funda records: {final.shape[0]}')
    return final.drop(['linkdt', 'linkenddt', 'gvkey'], axis=1)


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


def process_crsp_annual(filename,year_end=6):
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

    # aggregate up to annually
    data['rel'] = 1 + data.ret
    data.sort_values(['permno', 'date'], inplace=True)  # so that first cap below is correct
    data = data.groupby(['permno', 'year'], as_index=False).agg({'rel': 'prod', 'cap': 'first', 'siccd': 'first'})
    data['ret'] = data.rel - 1
    data.siccd = data.siccd.astype('str').str.zfill(4)
    data['sic1'] = data.siccd.str[:1]
    data['sic2'] = data.siccd.str[:2]
    data = data[['permno', 'sic1', 'sic2', 'year', 'ret', 'cap']].dropna()
    print(f'Annual records: {data.shape[0]}')
    return data


def process_crsp_new(filename, frequency='Q'):
    """
    allow quarterly and month
    """
    data = pd.read_feather(filename)
    data['share'] = data['shrout'] * data['cfacshr'] * 1e3
    data['price'] = data['prc'].abs() / data['cfacpr']
    data['cap'] = data.price * data.share / 1e6

    aggs = {'rel': 'prod', 'cap': 'last', 'siccd': 'last'}  # manual change
    info_cols = [k for k, v in aggs.items() if v == 'last']  # columns for beginnging quantities
    required_cols = ['date', 'permno', 'ret'] + info_cols  # for check

    if frequency == 'Q':
        time_delta = QuarterEnd
    elif frequency == 'M':
        time_delta = MonthEnd
    else:
        raise ValueError(f'Unrecognized frequency {frequency}')

    tota_rows = data.shape[0]
    print(f'Total rows: {tota_rows}')
    print(f'{data.date.min()} to {data.date.max()}')

    print('Check invalid returns')
    invalid = data[(data.ret == -66) | (data.ret == -77) | (data.ret == -88) | (data.ret == -99)]
    print(f'Invalid returns: {invalid.shape[0]}')

    print('Check missing')
    for col in required_cols:
        count = data[col].isna().sum()
        print(f'    {col} missing: {count} ({count / tota_rows:.2%})')

    data = data[required_cols].dropna()
    print(f'Remove NAs: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    data = data[(data.siccd < 6000) | (data.siccd > 6999)]  # remove financial
    print(f'Remove financials: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

    dup = data[['permno', 'date']].duplicated().sum()
    print(f'Duplicated for key [permno, date]: {dup}')

    data.permno = data.permno.astype(int)
    data['date'] = pd.to_datetime(data['date'])
    data['time_idx'] = data.date + time_delta(0)

    # aggregate up to annually
    data['rel'] = 1 + data.ret
    data.sort_values(['permno', 'date'], inplace=True)  # so that first cap below is correct
    data = data.groupby(['permno', 'time_idx'], as_index=False).agg(aggs)
    data['ret'] = data.rel - 1
    data.drop('rel', axis=1, inplace=True)

    # join with next quarter for forward looking returns
    data['time_idx_next'] = data.time_idx + time_delta(1)

    data = pd.merge(data, data[['permno', 'time_idx', 'ret']],
                    left_on=['permno', 'time_idx_next'], right_on=['permno', 'time_idx'],
                    suffixes=('', '_1'), how='inner')
    data.drop(['time_idx_next', 'time_idx_1'], axis=1, inplace=True)

    # get coarser industry classification
    data.siccd = data.siccd.astype('str').str.zfill(4)
    data['sic1'] = data.siccd.str[:1]
    data['sic2'] = data.siccd.str[:2]
    data.dropna(inplace=True)
    print(f'{frequency} records: {data.shape[0]}')
    return data


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


def process_compustat2(filename, year_end=12):
    fund = pd.read_feather(filename)
    total = fund.shape[0]
    print(f'Total rows: {total}')
    print(f'{fund.date.min()} to {fund.date.max()}')

    # unique permno, fyear
    fund = fund.sort_values(['permno', 'date'])
    fund = fund.groupby(['permno', 'fyear'], as_index=False).last()  # use the latest for each fiscal year

    critical_cols = ['at', 'che', 'act', 'lct', 'lt', 'sale']
    other_cols = ['dlc', 'dltt', 'ivao', 'ivst', 'oiadp', 'pstk']
    columns_kept = critical_cols + other_cols
    fund = fund[['permno', 'fyear'] + columns_kept].dropna(how='any', subset=critical_cols)
    fund = fund.fillna({'dltt': 0, 'dlc': 0, 'pstk': 0, 'ivst': 0, 'ivao': 0})
    print(f'Drop NAs: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

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

    cols = ['at', 'wc', 'nco', 'fin', 'sale']
    fund['fyear_prev'] = fund.fyear - 1
    join_keys = ['permno', 'fyear']
    fund = pd.merge(fund, fund[join_keys + cols],
                    left_on=['permno', 'fyear_prev'],
                    right_on=join_keys,
                    suffixes=['', '_prev'])
    fund.dropna(inplace=True)
    print(f'Merged with previous: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # delta metrics
    fund['avg_at'] = (fund['at'] + fund['at_prev']) / 2
    fund['roa'] = fund.oiadp / fund.avg_at
    fund['dwc'] = (fund.wc - fund.wc_prev) / fund.avg_at
    fund['dnco'] = (fund.nco - fund.nco_prev) / fund.avg_at
    fund['dnoa'] = fund['dwc'] + fund['dnco']
    fund['dfin'] = (fund.fin - fund.fin_prev) / fund.avg_at
    fund['tacc'] = fund.dwc + fund.dnco + fund.dfin
    fund.drop([col for col in fund.columns if col.endswith('_prev')], axis=1, inplace=True)
    print(f'Final rows: {fund.shape[0]} ({fund.shape[0] / total:.2%})')
    return fund


def process_instruction(df, instruction: str):
    [dest, formula] = [x.strip() for x in instruction.split('=')]


def process_compustat_new(fund, frequency='M'):
    """
    broad process function for compustat
    """
    if frequency == 'Q':
        time_delta = QuarterEnd
        steps = 4  # used to find previous annual record
    elif frequency == 'M':
        time_delta = MonthEnd
        steps = 12  # used to find previous annual record
    else:
        raise ValueError(f'Unrecognized frequency {frequency}')

    # dataframe schema
    required = ['at', 'che', 'act', 'lct', 'lt', 'sale']
    defaults = ['dlc', 'dltt', 'ivao', 'ivst', 'oiadp', 'pstk']  # per sloan 2005
    non_zeros = ['at']
    keep = ['dwc', 'dnco', 'dnoa', 'dfin', 'tacc', 'oa', 'oa1', 'dacy', 'dac1', 'dac2', 'dac3']

    total = fund.shape[0]
    print(f'Total rows: {total}')
    print(f'{fund.date.min()} to {fund.date.max()}')

    # unique permno, time idx
    fund['time_idx'] = fund.date + time_delta(0)
    fund = fund.sort_values(['permno', 'time_idx'])
    fund = fund.groupby(['permno', 'time_idx'], as_index=False).last()  # use the latest for each time idx

    # handle missing value
    fund.dropna(how='any', subset=required, inplace=True)
    fund = fund.fillna({col: 0 for col in defaults})
    print(f'Handle NAs: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # force non-zero on specified columns
    print('Check zeros')
    for col in non_zeros:
        zero = (fund[col] == 0).sum()
        print(f'    {col} has zeros: {zero}')
        fund = fund[fund[col] != 0]
        print(f'    Drop {col} zeros: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # ========== Before Join ==========
    # extended definition of accruals
    fund['coa'] = fund.act - fund.che
    fund['col'] = fund.lct - fund.dlc
    fund['wc'] = fund.coa - fund.col
    fund['ncoa'] = fund['at'] - fund.act - fund.ivao
    fund['ncol'] = fund['lt'] - fund.lct - fund.dltt
    fund['nco'] = fund.ncoa - fund.ncol
    fund['fina'] = fund.ivst + fund.ivao
    fund['finl'] = fund.dltt + fund.dlc + fund.pstk
    fund['fin'] = fund.fina - fund.finl

    # ========== Join ==========
    # assume we always need some items from previous period
    # TODO: automate this previou join step
    # a lot of the quantities are based on delta, which is sensitive to time delta between now and the previous record
    # if the fiscal year end of a company is changed such that there are only 6 month between this end and the last end,
    # then we shouldn't use this record. Hopefully, we won't discard too much in this way
    fund['time_idx_p'] = fund.time_idx - time_delta(steps)  # require 12 month apart
    fund = pd.merge(fund, fund, left_on=['permno', 'time_idx_p'], right_on=['permno', 'time_idx'], suffixes=['', '_p'])
    # fund.dropna(inplace=True)  # don't drop na here because some cols with missing values are not needed
    print(f'After joining previous: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # ========== After Join ==========
    # operating accruals
    fund['dca'] = fund.act - fund.act_p
    fund['dcash'] = fund.che - fund.che_p
    fund['dcl'] = fund.lct - fund.lct_p
    fund['dstd'] = fund.dlc - fund.dlc_p
    fund['dtp'] = (fund.txp - fund.txp_p).fillna(0)  # set to 0 if missing
    fund['oa'] = (fund.dca - fund.dcash) - (fund.dcl - fund.dstd - fund.dtp) - fund.dp
    fund['oa1'] = fund.ni - fund.oancf
    fund.loc[fund.fyear <= 1989, 'oa1'] = fund.loc[fund.fyear <= 1989, 'oa']

    # DAC
    fund['dsale'] = fund.sale - fund.sale_p
    fund['drec'] = fund.rect - fund.rect_p
    fund['dacy'] = fund.oa / fund.at_p
    fund['dac1'] = 1 / fund.at_p
    fund['dac2'] = (fund.dsale - fund.drec) / fund.at_p
    fund['dac3'] = fund.ppegt / fund.at_p

    # extended defintion of accruals
    fund['avg_at'] = (fund['at'] + fund['at_p']) / 2
    fund['dwc'] = (fund.wc - fund.wc_p) / fund.avg_at
    fund['dnco'] = (fund.nco - fund.nco_p) / fund.avg_at
    fund['dnoa'] = fund.dwc + fund.dnco
    fund['dfin'] = (fund.fin - fund.fin_p) / fund.avg_at
    fund['tacc'] = fund.dwc + fund.dnco + fund.dfin

    fund = fund[['permno', 'time_idx'] + keep].dropna()
    print(f'Final rows: {fund.shape[0]} ({fund.shape[0] / total:.2%})')
    return fund


def lag(df, col, lag=12):
    return df.groupby(level=0)[col].shift(lag)


def process_compustat_sentinel(fund):
    """
    broad process function for compustat
    """
    # dataframe schema
    required = ['at', 'che', 'act', 'lct', 'lt', 'sale']
    defaults = ['dlc', 'dltt', 'ivao', 'ivst', 'oiadp', 'pstk']  # per sloan 2005
    non_zeros = ['at']
    keep = ['dwc', 'dnco', 'dnoa', 'dfin', 'tacc', 'oa', 'oa1', 'dacy', 'dac1', 'dac2', 'dac3']

    total = fund.shape[0]
    print(f'Total rows: {total}')
    print(f'{fund.date.min()} to {fund.date.max()}')

    # unique permno, time idx
    fund['time_idx'] = fund.date + MonthEnd(0)
    fund = fund.sort_values(['permno', 'time_idx'])
    fund = fund.groupby(['permno', 'time_idx'], as_index=False).last()  # use the latest for each time idx

    # handle missing value
    fund.dropna(how='any', subset=required, inplace=True)
    fund = fund.fillna({col: 0 for col in defaults})
    print(f'Handle NAs: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # force non-zero on specified columns
    print('Check zeros')
    for col in non_zeros:
        zero = (fund[col] == 0).sum()
        print(f'    {col} has zeros: {zero}')
        fund = fund[fund[col] != 0]
        print(f'    Drop {col} zeros: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    fund = fund[fund.time_idx > '1970-01-01']
    print(f'After 1970: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # ========== Before Join ==========
    # extended definition of accruals
    fund['coa'] = fund.act - fund.che
    fund['col'] = fund.lct - fund.dlc
    fund['wc'] = fund.coa - fund.col
    fund['ncoa'] = fund['at'] - fund.act - fund.ivao
    fund['ncol'] = fund['lt'] - fund.lct - fund.dltt
    fund['nco'] = fund.ncoa - fund.ncol
    fund['fina'] = fund.ivst + fund.ivao
    fund['finl'] = fund.dltt + fund.dlc + fund.pstk
    fund['fin'] = fund.fina - fund.finl

    # ========== Use sentinel ==========
    # to allow monthly record. not the most efficiency way. But trade time for accuracy
    start = time.time()
    sentinel = []
    whole_range = pd.date_range(fund.time_idx.min(), fund.time_idx.max(), freq='m')
    whole_range = pd.DataFrame({'date': whole_range}, index=whole_range)

    time_range = fund.groupby('permno').agg({'time_idx': ['min', 'max']})
    for permno, times in time_range['time_idx'].iterrows():
        dates = whole_range.loc[times['min']: times['max']].values.flatten()
        sentinel.append(pd.DataFrame({'time_idx': dates, 'permno': permno}))
    sentinel = pd.concat(sentinel, axis=0)
    print(time.time() - start, 's')

    fund = pd.merge(fund, sentinel, on=['time_idx', 'permno'], how='outer')
    fund = fund.set_index(['permno', 'time_idx']).sort_index()
    fund = fund.groupby(level=0).fillna(method='ffill')
    total = fund.shape[0]
    print(f'Expended rows: {total}')

    # ========== After Join ==========
    # operating accruals
    fund['dca'] = fund.act - lag(fund, 'act')
    fund['dcash'] = fund.che - lag(fund, 'che')
    fund['dcl'] = fund.lct - lag(fund, 'lct')
    fund['dstd'] = fund.dlc - lag(fund, 'dlc')
    fund['dtp'] = (fund.txp - lag(fund, 'txp')).fillna(0)  # set to 0 if missing
    fund['oa'] = (fund.dca - fund.dcash) - (fund.dcl - fund.dstd - fund.dtp) - fund.dp
    fund['oa1'] = fund.ni - fund.oancf
    fund.loc[fund.fyear <= 1989, 'oa1'] = fund.loc[fund.fyear <= 1989, 'oa']

    # DAC
    fund['dsale'] = fund.sale - lag(fund, 'sale')
    fund['drec'] = fund.rect - lag(fund, 'rect')
    fund['dacy'] = fund.oa / lag(fund, 'at')
    fund['dac1'] = 1 / lag(fund, 'at')
    fund['dac2'] = (fund.dsale - fund.drec) / lag(fund, 'at')
    fund['dac3'] = fund.ppegt / lag(fund, 'at')

    # extended defintion of accruals
    fund['avg_at'] = (fund['at'] + lag(fund, 'at')) / 2
    fund['dwc'] = (fund.wc - lag(fund, 'wc')) / fund.avg_at
    fund['dnco'] = (fund.nco - lag(fund, 'nco')) / fund.avg_at
    fund['dnoa'] = fund.dwc + fund.dnco
    fund['dfin'] = (fund.fin - lag(fund, 'fin')) / fund.avg_at
    fund['tacc'] = fund.dwc + fund.dnco + fund.dfin

    fund = fund[keep].dropna(subset=['oa'])
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