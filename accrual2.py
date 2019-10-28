
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
    """
    difference of two dataframe by keys
    """
    if right is None:
        right = left

    df2 = df2[right].copy()
    df2['_flag_'] = 1
    joined = pd.merge(df1, df2, left_on=left, right_on=right, how='left', suffixes=('', '_y'))
    joined = joined[joined['_flag_'].isna()]
    return joined.drop([col for col in joined.columns if col.endswith('_y')] + ['_flag_'], axis=1)


def lag(df, col, lag=12):
    return df.groupby(level=0)[col].shift(lag)


def calculate_IC(df, ret_col, time_col, alphas):
    """
    calculate information coefficient
    """
    skip = len(alphas) + 1
    ic = df.groupby(time_col)[alphas + [ret_col]].corr(method='kendall').iloc[(skip - 1)::skip]
    ic = ic.reset_index().drop(['level_1', ret_col], axis=1)
    return ic


def calculate_IC_decay(df, signals, reverse=None):
    """
    calculate information coefficient decay
    """
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


def append_forward_returns(raw, periods, ret_col='ret'):
    """
    append forward loocking returns. For the use of IC decay
    """
    data = raw.dropna(subset=[ret_col]).copy()
    data = data.set_index(['permno', 'time_idx']).sort_index()
    for period in periods:
        data['ret_' + str(period)] = data.groupby(level=0)[ret_col].shift(-period)
    return data.dropna()


def append_past_returns(raw, periods):
    """
    append past return. For the use of momentum / priced effect
    """
    data = raw.dropna(subset=['ret']).copy()
    wide = data.pivot(index='time_idx', columns='permno', values='ret')
    wide = wide + 1

    rets = []
    for period in periods:
        tmp = wide.rolling(window=period).apply(np.prod, raw=True) - 1
        tmp = tmp.reset_index().melt(id_vars='time_idx', var_name='permno', value_name='prev_ret_' + str(period)).dropna()
        tmp.permno = tmp.permno.astype('int')
        tmp.time_idx = pd.to_datetime(tmp.time_idx)
        rets.append(tmp)

    for ret in rets:
        data = pd.merge(data, ret, on=['time_idx', 'permno'], how='left')
    return data  # don't drop because we may need to work on other fundamental data too


def calculate_hedeged_return_by_industry(df, factor, long, short, ind_col='sic1', ret_col='ret_1', ncuts=3):
    """
    hedged return per sector, long 50% short 50%. for sector analysis
    """
    df = df.copy()

    # sorting
    df[factor] = df.groupby(['time_idx', ind_col], as_index=False)[factor].transform(lambda x: pd.qcut(x, ncuts, range(ncuts)))

    # calculate bucket returns
    port_ret = df.groupby(['time_idx', ind_col, factor], as_index=False).agg({ret_col: 'mean', 'permno': 'count'})
    summary = port_ret.set_index(factor).groupby(['time_idx', ind_col], as_index=False).apply(
        lambda x: pd.Series([
            (x.loc[long, ret_col] - x.loc[short, ret_col]) / 2,
            (x.loc[long, 'permno'] + x.loc[short, 'permno'])
        ], index=['ret', 'count'])
    )

    summary = summary.groupby(level=1).agg({'ret': ['mean', 'std'], 'count': 'mean'})
    summary.columns = ["_".join(x) if x[1] != '' else x[0] for x in summary.columns.ravel()]
    summary['t stat'] = summary.ret_mean / summary.ret_std
    summary = summary.rename({'ret_mean': 'return', 'count_mean': 'avg count'}, axis=1).drop('ret_std', axis=1)
    return summary.reset_index()[[ind_col, 'return', 't stat', 'avg count']]


def calculate_industry_neutral_nway_return(df, name, sorts, cuts, long_bucket, short_bucket, ind_col='sic1', ret_col='ret'):
    """
    calculate long short return controlled for sector (or size)

    :param df: input dataframe
    :param name: factor name, just for naming the output
    :param sorts: factors to be sorted. The nth factor will be sorted such that it's independent of the first n-1 factors
    :param cuts: number of buckets of each sorts
    :param long_bucket: bucket to long
    :param short_bucket: bucket to short
    :param ind_col: controlling variable, like sector or size bucket
    :param ret_col: forward looking return column
    :return: structure of equity curve and related statistics
    """
    df = df.copy()

    # n-way sort
    keys = ['time_idx', ind_col]
    for sort, cut in zip(sorts, cuts):
        df[sort] = df.groupby(keys, as_index=False)[sort].transform(lambda x: pd.qcut(x, cut, range(cut)))
        keys.append(sort)

    # calculate bucket returns
    port_ret = df.groupby(['time_idx', ind_col] + sorts, as_index=False).agg({ret_col: 'mean', 'permno': 'count'})
    summary = port_ret.groupby(['time_idx'] + sorts, as_index=False)[[ret_col, 'permno']].apply(
        lambda x: pd.Series([np.average(x[ret_col], weights=x.permno), x.permno.sum()], index=['ret', 'avg count'])
    )

    hedged_bucket = summary.reset_index('time_idx').groupby('time_idx').apply(
        lambda x: pd.Series([
            (x.loc[long_bucket, 'ret'] - x.loc[short_bucket, 'ret']) / 2,
            (x.loc[long_bucket, 'avg count'] + x.loc[short_bucket, 'avg count']) / 2
        ], index=['ret', 'avg count'])
    )
    tmp = hedged_bucket.agg({'ret': ['mean', 'std'], 'avg count': 'mean'})
    hedged_bucket = pd.DataFrame({**{sorts[0]: ['hedged'],
                                     'return': [tmp.loc['mean', 'ret']],
                                     't stat': [tmp.loc['mean', 'ret'] / tmp.loc['std', 'ret']],
                                     'avg count': [tmp.loc['mean', 'avg count']]}, **{x: '' for x in sorts[1:]}})

    summary = summary.groupby(sorts).agg({'ret': ['mean', 'std'], 'avg count': 'mean'}, axis=1)
    summary.columns = ["_".join(x) if x[1] != '' else x[0] for x in summary.columns.ravel()]
    summary['t stat'] = summary.ret_mean / summary.ret_std
    summary = summary.drop('ret_std', axis=1).rename({'ret_mean': 'return', 'avg count_mean': 'avg count'}, axis=1)
    summary = summary.reset_index()[sorts + ['return', 't stat', 'avg count']]
    summary = pd.concat([summary, hedged_bucket], axis=0, sort=False)

    # calculate hedged returns
    agg = port_ret.set_index(sorts).groupby(['time_idx', ind_col]).apply(
        lambda x: pd.Series([
            (x.loc[long_bucket, ret_col] - x.loc[short_bucket, ret_col]) / 2,  # to maintain unit leverage
            (x.loc[long_bucket, ret_col]) / 2,  # division by 2 to match same leverage of the long side
            (x.loc[short_bucket, ret_col]) / 2,  # division by 2 to match same leverage of the short side
            (x.loc[long_bucket, 'permno'] + x.loc[short_bucket, 'permno']) / 2
        ], index=['ret', 'long_ret', 'short_ret', 'count'])
    )
    hedged = agg.groupby(level=0).apply(lambda x: np.average(x.ret, weights=x['count']))
    long = agg.groupby(level=0).apply(lambda x: np.average(x.long_ret, weights=x['count']))
    short = agg.groupby(level=0).apply(lambda x: np.average(x.short_ret, weights=x['count']))

    # plot
    avg = hedged.rolling(4, min_periods=4).sum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(y=avg, x=avg.index, name=name + ' Return'), secondary_y=False)
    fig.add_trace(go.Scatter(y=(1 + hedged).cumprod(), x=hedged.index, name=name + ' Equity'), secondary_y=True)
    fig.add_trace(go.Scatter(y=(1 + long).cumprod(), x=hedged.index, name=name + ' Long Equity'), secondary_y=True)
    fig.add_trace(go.Scatter(y=(1 + short).cumprod(), x=hedged.index, name=name + ' Short Equity'), secondary_y=True)
    fig.show()

    return {'name': name, 'equity': hedged, 'summary': summary, 'data': df}


def plot_comparison(results):
    """
    plot multiple equity curve
    :param results: output from function "calculate_industry_neutral_nway_return"
    """
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


def get_funda_data(conn, items):
    """
    download Compustat data from WRDS
    """
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
    preprocessing steps for CRSP dataset
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
    # data = data[(data.siccd < 1500) | (data.siccd > 1799)]  # remove construction
    # data = data[data.siccd < 9100]  # remove construction
    print(f'Remove industries: {data.shape[0]} ({data.shape[0] / tota_rows:.2%})')

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
    preprocessing step for compustat, including factor generation
    """
    # dataframe schema
    required = ['at', 'che', 'act', 'lct', 'lt', 'sale']
    defaults = ['dlc', 'dltt', 'ivao', 'ivst', 'oiadp', 'pstk']  # per sloan 2005
    non_zeros = ['at']
    keep = ['dwc', 'dnco', 'dnoa', 'dfin', 'tacc', 'tacc2', 'oa', 'dacy', 'dac1', 'dac2', 'dac3']

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
    fund['avg_at'] = (fund['at'] + lag(fund, 'at')) / 2
    fund['dsale'] = fund.sale - lag(fund, 'sale')
    fund['drec'] = fund.rect - lag(fund, 'rect')
    fund['dacy'] = fund.oa / fund.avg_at
    fund['dac1'] = 1 / fund.avg_at
    fund['dac2'] = (fund.dsale - fund.drec) / fund.avg_at
    fund['dac3'] = fund.ppegt / fund.avg_at

    # extended defintion of accruals
    fund['dwc'] = (fund.wc - lag(fund, 'wc')) / fund.avg_at
    fund['dnco'] = (fund.nco - lag(fund, 'nco')) / fund.avg_at
    fund['dnoa'] = fund.dwc + fund.dnco
    fund['dfin'] = (fund.fin - lag(fund, 'fin')) / fund.avg_at
    fund['tacc'] = fund.dwc + fund.dnco + fund.dfin
    fund['tacc2'] = fund.dwc + fund.dnco - fund.dfin

    fund = fund[keep].dropna().reset_index()
    print(f'Final rows: {fund.shape[0]} ({fund.shape[0] / total:.2%})')
    return fund


def append_dac(panel, ind_col='industry'):
    """
    calculate discretionary accrual
    """
    dac = []
    for (t, s), data in panel.groupby(['time_idx', ind_col]):
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
