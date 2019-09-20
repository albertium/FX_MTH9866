
import pandas as pd
import numpy as np

# data = pd.read_csv('data/crsp_monthly.gz', compression='gzip', parse_dates=['date'], infer_datetime_format='%Y%m%d',
#                    dtype={'SICCD': str, 'PERMNO': str, 'PERMCO': str, 'PRC': str, 'RET': str, 'CFACPR': str})
# data.SICCD = data.SICCD.replace('Z', np.nan)
# data.RET = data.RET.replace('C', np.nan)
# data = data.dropna()
# data = data.astype({'SICCD': 'int32', 'RET': 'float64'})
# data = data[(data.SICCD < 6000) | (data.SICCD > 6999)]  # remove financial
# data.to_pickle('data/final.pkl')

data = pd.read_pickle('data/final.pkl')
np.savetxt('data/code.csv', data.PERMNO.unique(), delimiter=',')