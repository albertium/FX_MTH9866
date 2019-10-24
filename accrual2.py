
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd, QuarterEnd, YearEnd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
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


def lag(df, col, lag=12):
    return df.groupby(level=0)[col].shift(lag)


def neutralize_alphas_and_returns(df, group_by, ret_col, alphas):
    def normalize(x):
        normal = (x - x.mean()) / x.std()
        return normal.clip(-3, 3)

    df = df.copy()
    # neutralize alphas
    df[alphas] = df.groupby(group_by, as_index=False)[alphas].transform(normalize)

    # neutralize returns
    df = df.set_index(group_by + ['permno'])
    mean = df.groupby(level=list(range(len(group_by))))[ret_col].mean()
    df['adj_ret'] = df[ret_col] - mean
    return df.dropna().reset_index()


def calculate_IC(df, ret_col, time_col, alphas):
    skip = len(alphas) + 1
    ic = df.groupby(time_col)[alphas + [ret_col]].corr(method='kendall').iloc[(skip - 1)::skip]
    ic = ic.reset_index().drop(['level_1', ret_col], axis=1)
    return ic


def calculate_IC_decay(df, signals, reverse=None):
    df = df.set_index('time_idx')
    ret_cols = [x for x in df.columns if x.startswith('ret_')]
    periods = [int(x.replace('ret_', '')) for x in ret_cols]

    if reverse is None:
        reverse = {}
    else:
        reverse = {k: 1 for k in reverse}

    data = {}
    for signal in signals:
        sign = 1 if signal in reverse else -1
        corr = []
        for col in ret_cols:
            tmp = df[[signal, col]].dropna().groupby(level=0).corr(method='kendall') * sign
            corr.append(tmp.groupby(level=0).first()[col].mean())
        data[signal] = corr

    data['period'] = periods
    return pd.DataFrame(data).set_index('period')


def append_forward_returns(raw, periods):
    data = raw.dropna(subset=['ret']).copy()
    wide = data.pivot(index='time_idx', columns='permno', values='ret')
    wide = wide + 1

    rets = []
    for period in periods:
        tmp = wide.rolling(window=period).apply(np.prod, raw=True).shift(-period) - 1
        tmp = tmp.reset_index().melt(id_vars='time_idx', var_name='permno', value_name='ret_' + str(period)).dropna()
        tmp.time_idx = pd.to_datetime(tmp.time_idx)
        rets.append(tmp)

    for ret in rets:
        data = pd.merge(data, ret, on=['time_idx', 'permno'], how='left')
    return data.dropna()


def append_past_returns(raw, periods):
    data = raw.dropna(subset=['ret']).copy()
    wide = data.pivot(index='time_idx', columns='permno', values='ret')
    wide = wide + 1

    rets = []
    for period in periods:
        tmp = wide.rolling(window=period).apply(np.prod, raw=True) - 1
        tmp = tmp.reset_index().melt(id_vars='time_idx', var_name='permno', value_name='prev_ret_' + str(period)).dropna()
        tmp.time_idx = pd.to_datetime(tmp.time_idx)
        rets.append(tmp)

    for ret in rets:
        data = pd.merge(data, ret, on=['time_idx', 'permno'], how='left')
    return data  # don't drop because we may need to work on other fundamental data too


def calculate_nway_sorted_return(df, name, sorts, cuts, long_bucket, short_bucket, ret_col='ret'):
    df = df.copy()

    # n-way sort
    keys = ['time_idx']
    for sort, cut in zip(sorts, cuts):
        df[sort] = df.groupby(keys, as_index=False)[sort].transform(lambda x: pd.qcut(x, cut, range(cut)))
        keys.append(sort)

    # calculate returns
    port_ret = df.groupby(['time_idx'] + sorts, as_index=False)[ret_col].mean()
    summary = port_ret.groupby(sorts, as_index=False)[ret_col].mean()
    hedged = port_ret.set_index(sorts).groupby('time_idx')\
        .apply(lambda x: (x.loc[long_bucket, ret_col] - x.loc[short_bucket, ret_col]) / 2)

    # plot
    # avg = hedged.rolling(4, min_periods=4).sum()
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(go.Bar(y=avg, x=avg.index, name=name + ' Return'), secondary_y=False)
    # fig.add_trace(go.Scatter(y=(1 + hedged).cumprod(), x=hedged.index, name=name + ' Equity'), secondary_y=True)
    # fig.show()

    return {'name': name, 'equity': hedged, 'summary': summary}


def plot_comparison(results):
    dfs = []
    for res in results:
        equity = (1 + res['equity']).cumprod()
        equity.name = 'equity'
        equity = equity.reset_index()
        equity['name'] = res['name']
        dfs.append(equity)
    data = pd.concat(dfs, axis=0)

    fig = px.line(data, x='time_idx', y='equity', color='name')
    fig.show()


def calculate_sorted_returns(df, signals, time_col='year', ret_col='ret', weight_col='cap', n_cuts=10, reverse=None):
    df = df.copy()
    cut_idx = list(range(n_cuts))
    df['equal'] = 1

    if reverse is None:
        reverse = {}
    else:
        reverse = {k: 1 for k in reverse}

    for signal in signals:
        df[signal] = df.groupby(time_col, as_index=False)[signal].transform(lambda x: pd.qcut(x, n_cuts, cut_idx))

    results = []
    df['scaled_ret'] = df[weight_col] * df[ret_col]
    for rank in signals:
        port_ret = df.groupby([rank, time_col], as_index=False).agg({'scaled_ret': 'sum', weight_col: 'sum'})
        port_ret['wgt_ret'] = port_ret.scaled_ret / port_ret[weight_col]

        port_ret = port_ret.pivot(index=time_col, columns=rank, values='wgt_ret').reset_index()
        if rank in reverse:
            port_ret['hedged'] = (port_ret.iloc[:, -1] - port_ret.iloc[:, 1]) / 2  # make sure leverage is 1
        else:
            port_ret['hedged'] = (port_ret.iloc[:, 1] - port_ret.iloc[:, -1]) / 2  # make sure leverage is 1
        port_ret['avg'] = port_ret.hedged.rolling(4, min_periods=4).sum()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(y=port_ret.avg, x=port_ret[time_col], name=rank), secondary_y=False)
        fig.add_trace(go.Scatter(y=(1 + port_ret.hedged).cumprod(), x=port_ret[time_col], name='Equity'), secondary_y=True)
        fig.show()
        results.append(port_ret)

    return results


def calculate_returns_by_buckets(df, signals, time_col='year', ret_col='ret', weight_col='cap', n_cuts=10):
    df = df.copy()
    cut_idx = list(range(n_cuts))
    df['equal'] = 1

    # set up exempt list
    for signal in signals:
        df[signal] = df.groupby(time_col, as_index=False)[signal].transform(lambda x: pd.qcut(x, n_cuts, cut_idx))

    results = []
    df['scaled_ret'] = df[weight_col] * df[ret_col]
    df['hit'] = df[ret_col] > 0  # for calculating hit rate
    for rank in signals:
        port_ret = df.groupby([rank, time_col], as_index=False)\
            .agg({'scaled_ret': 'sum', weight_col: 'sum', 'hit': 'mean', ret_col: 'count'})
        port_ret['wgt_ret'] = port_ret.scaled_ret / port_ret[weight_col]

        # organize result
        overall = port_ret.groupby(rank, as_index=False)\
            .agg({'wgt_ret': ['mean', 'std'], 'hit': ['mean', 'count'], ret_col: 'mean'}, axis=1)
        overall.columns = ["_".join(x) if x[1] != '' else x[0] for x in overall.columns.ravel()]
        overall['t_stat'] = overall.wgt_ret_mean / overall.wgt_ret_std
        overall = overall.set_index(rank).drop('wgt_ret_std', axis=1)\
            .rename({'wgt_ret_mean': 'ret', 'hit_mean': 'hit', 'hit_count': 'count', ret_col + '_mean': 'size'}, axis=1)
        results.append(overall[['ret', 't_stat', 'hit', 'count', 'size']])

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


def process_crsp(filename, frequency='Q'):
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

    # aggregate up to quarterly
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


def process_compustat(fund):
    """
    broad process function for compustat
    """
    # dataframe schema
    required = ['at', 'che', 'act', 'lct', 'lt', 'sale']
    defaults = ['dlc', 'dltt', 'ivao', 'ivst', 'oiadp', 'pstk']  # per sloan 2005
    non_zeros = ['at']
    keep = ['dwc', 'dnco', 'dnoa', 'dfin', 'tacc', 'oa', 'dacy', 'dac1', 'dac2', 'dac3']

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
    fund['oa'] = ((fund.dca - fund.dcash) - (fund.dcl - fund.dstd - fund.dtp) - fund.dp) / lag(fund, 'at')

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

    fund = fund[keep].dropna().reset_index()
    print(f'Final rows: {fund.shape[0]} ({fund.shape[0] / total:.2%})')
    return fund


def append_dac(panel):
    dac = []
    for (t, s), data in panel.groupby(['time_idx', 'sic2']):
        data = data[['permno', 'dacy', 'dac1', 'dac2', 'dac3']].dropna()
        if not data.empty:
            res = sm.OLS(data.dacy, data[['dac1', 'dac2', 'dac3']]).fit().resid
            df = pd.DataFrame({'permno': data.permno, 'dac': res})
            df['time_idx'] = t
            dac.append(df)

    dac = pd.concat(dac, axis=0, ignore_index=True)
    print(f'Before join: {panel.shape[0]}')
    df = pd.merge(panel, dac, on=['permno', 'time_idx'], how='inner').dropna()
    print(f'After join: {panel.shape[0]}')
    return df.drop(['dacy', 'dac1', 'dac2', 'dac3'], axis=1)
