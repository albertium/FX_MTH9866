import numpy as np
from scipy.stats import norm


sig = 0.0875
r = 0.0175
Q = 0.0175 - np.log(1.004) * 2
print(1.004 * np.exp(sig ** 2 / 2 * 0.5 - sig * np.sqrt(0.5) * norm.ppf(0.25 * np.exp(Q * 0.5))))