from numpy import tile, mean, dot, diag, sqrt
from scipy.stats import t

def auto_cov(ts, lag=0, ddof=0):
	"""
	calculate auto covariance of time series ts
	"""
	T = ts.shape[0]
	means = mean(ts, axis=0)
	if len(ts.shape) == 1:
		ts = ts - means
	else:
		ts = ts - tile(means, [T,1])
	return 1./(T-ddof)*dot(ts[:T-lag, ].T, ts[lag:, ])

def newey_west(ts, lag=4):
	"""
	calculate  newey west 
	"""
	S = auto_cov(ts)
	if lag > 0:
		for i in range(1, lag+1):
			j_lagged_auto = auto_cov(ts, lag=i)
			S += (1. - i/(lag+1.))*(j_lagged_auto + j_lagged_auto.T)
	return S

def sample_cov(x, y, ddof=0):
	T_x = x.shape[0]
	means = mean(x, axis=0)
	if len(x.shape) == 1:
		x = x - means
	else:
		x = x - tile(means, [T_x,1])
	means = mean(y, axis=0)
	if len(y.shape) == 1:
		y = y - means
	else:
		y = y - tile(means, [T_x,1])
	return 1./(T_x-ddof)*dot(x.T , y)

def t_test(estimate, cov, nobs):
	tstats = estimate/sqrt(1./nobs*diag(cov))
	significance = [1- t.cdf(abs(each), df=nobs) for each in tstats]
	return tstats, significance