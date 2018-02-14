import numpy as np

def auto_cov(ts, lag=0, ddof=0):
	"""
	calculate auto covariance of time series ts
	"""
	T = ts.shape[0] - ddof
	mean = np.mean(ts, axis=0)
	ts = ts - np.tile(mean, [T,1])
	return 1./T*np.inner(ts[:T-lag, :].T, ts[lag:, :].T)

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
	T_x = x.shape[0] - ddof
	mean = np.mean(x, axis=0)
	x = x - np.tile(mean, [T_x,1])
	mean = np.mean(y, axis=0)
	y = y - np.tile(mean, [T_x,1])
	return 1./T_x*inner(x.T , y.T)