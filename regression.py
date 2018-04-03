from base import BaseRegressionEstimator
from numpy import dot, vstack, ones, var, sqrt, diag, repeat, mean, vectorize, hstack, identity, array, ravel, concatenate
from numpy.linalg import inv
from scipy.stats import t
from scipy.linalg import block_diag

class OLS(BaseRegressionEstimator):
	def __init__(self, X, y, robust=False, intercept=True, test_statistics=True, demeaned=False):
		"""
		X is exogenous variable
		y is endogeous variable
		"""
		self.X = X
		self.y = y
		self._nobs = len(y)
		self._robust = robust
		self._intercept = intercept
		self._demeaned = demeaned
		self._nparam = X.shape[1]
		if self._demeaned:
			self.X = self.X - repeat(mean(self.X, axis=0), self._nobs).reshape(self.X.shape)
			self.y = self.y - repeat(mean(self.y, axis=0), self._nobs).reshape(self.y.shape)
		elif self._intercept:
			## the first column is always intercept
			self.X = vstack((ones(self._nobs), self.X.T)).T
			self._nparam += 1
		self._coef = {}
		self._test = test_statistics
		self._ddof = self._nobs - self._nparam
		self._res = None
		self._var = None

	def _run_test(self, invSxx, Sxx):
		if self._robust:
			self._coef['var'] = dot(dot(invSxx, dot(dot(self.X.T, diag(self._res**2)), self.X)), invSxx)
		else:
			self._coef['var'] = self._var*inv(Sxx)
		self._coef['t-test'] = self._coef['beta']/sqrt(diag(self._coef['var']))
		self._coef['significance'] = 1- t.cdf(abs(self._coef['t-test']), df=self._ddof)

	def fit(self):
		Sxx = dot(self.X.T, self.X)
		Sxy = dot(self.X.T, self.y)
		invSxx = inv(Sxx)
		self._coef['beta'] = dot(invSxx, Sxy)
		self._res = self.y - dot(self._coef['beta'], self.X.T)
		self._var = var(self._res, ddof=self._ddof)
		if self._test:
			self._run_test(invSxx, Sxx)
			
	def predict(self, X):
		if self._intercept:
			if len(X.shape) == 1:
				X = hstack((np.ones(1), X))
			else:
				X = vstack((np.ones(X.shape[0]), X.T)).T
		return dot(self._coef['beta'], X.T)

class PooledRegression(BaseRegressionEstimator):
	def __init__(self, X, y, Z=None, random_effect=False, test_statistics=False):
		"""
		X is time variant 3-D array
		Z is time invariant 3-D array
		y is target 3-D array
		"""

		if random_effect:
			X = array([vstack((ones(each.shape[0]), each.T)).T for each in X])
		if len(X[0].shape) == 1:
			self._nparam = X.shape[0]
		else:
			self._nparam = X[0].shape[1]

		self._nentity = X.shape[0]

		if Z is None:
			self.X = block_diag(*X)
		else:
			self.X = block_diag(*concatenate((X, Z), axis=2))
		self.y = hstack(y)
		self._nobs = len(self.y)
		self._random_effect = random_effect
		self.X_demeaned = block_diag(*array([self._demean(each) for each in X]))
		self.y_demeaned = hstack(array([self._demean(each) for each in y]))
		self._k = dot(inv(dot(self.X.T, self.X)), dot(self.X_demeaned.T, self.X_demeaned))
		self._coef = {}
		self._test = test_statistics

	def _demean(self, X):
		X = X - repeat(mean(X, axis=0), X.shape[0]).reshape(X.shape)
		return X
	
	def fit(self):
		pooled = OLS(self.X, self.y, intercept=False, test_statistics=self._test)
		pooled.fit()
		self._coef['pooled'] = pooled.get('_coef')
		within = OLS(self.X_demeaned, self.y_demeaned, intercept=False, test_statistics=self._test)
		within.fit()
		self._coef['within'] = within.get('_coef')
		self._coef['between'] = {'beta': dot(inv(identity(self._nparam*self._nentity)-self._k), 
								self._coef['pooled']['beta'] - dot(self._k, self._coef['within']['beta']))}

	def predict(self, X):
		pass
		
