
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def join_crsp_and_funda(crsp, funda):
    crsp['year_p'] = crsp.year - 1
    joined = pd.merge(crsp, funda, left_on=['permno', 'year_p'], right_on=['permno', 'fyear'], how='inner')
    print(f'CRSP recrods: {crsp.shape[0]}')
    print(f'Merged recrods: {joined.shape[0]} ({joined.shape[0] / crsp.shape[0]:.2%})')
    return joined


def neutralize_alphas_and_returns(df, group_by, alphas):
    def normalize(x):
        normal = (x - x.mean()) / x.std()
        return np.maximum(np.minimum(normal, 3), -3)

    df = df.copy()
    # neutralize alphas
    df[[x + '_n' for x in alphas]] = df.groupby(group_by, as_index=False)[alphas].transform(normalize)

    # neutralize returns
    industry_rets = df.groupby(group_by, as_index=False).ret.mean()
    df = pd.merge(df, industry_rets, on=group_by, suffixes=['', '_industry'])
    df['adj_ret'] = df.ret - df.ret_industry
    return df.dropna()


def calculate_IC(df, time_col, alphas):
    skip = len(alphas) + 1
    ic = df.groupby(time_col)[alphas + ['adj_ret']].corr(method='spearman').iloc[(skip - 1)::skip]
    ic = ic.reset_index().drop(['level_1', 'adj_ret'], axis=1)
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
        port_ret['hedged'] = port_ret[0] - port_ret[9]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(y=port_ret.hedged, x=port_ret.year, name='Annual Return'), secondary_y=False)
        fig.add_trace(go.Scatter(y=(1 + port_ret.hedged).cumprod(), x=port_ret.year, name='Equity'), secondary_y=True)
        fig.show()
        results.append(port_ret)

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


def process_compustat_new(fund):
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

    # unique permno, fyear
    fund = fund.sort_values(['permno', 'date'])
    fund = fund.groupby(['permno', 'fyear'], as_index=False).last()  # use the latest for each fiscal year

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
    join_keys = ['permno', 'fyear']
    fund['fyear_p'] = fund.fyear - 1
    fund = pd.merge(fund, fund, left_on=['permno', 'fyear_p'], right_on=join_keys, suffixes=['', '_p'])
    # fund.dropna(inplace=True)
    # print(f'After joining previous: {fund.shape[0]} ({fund.shape[0] / total:.2%})')

    # ========== After Join ==========
    # operating accruals
    fund['dca'] = fund.act - fund.act_p
    fund['dcash'] = fund.che - fund.che_p
    fund['dcl'] = fund.lct - fund.lct_p
    fund['dstd'] = fund.dlc - fund.dlc_p
    fund['dtp'] = (fund.txp - fund.txp_p).fillna(0)  # set to 0 if missing
    fund['oa'] = (fund.dca - fund.dcash) - (fund.dcl - fund.dstd - fund.dtp) - fund.dp

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

    fund = fund[['permno', 'fyear'] + keep].dropna()
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