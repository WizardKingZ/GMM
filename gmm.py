from numpy import dot, ones, identity, mean
from numpy.linalg import inv
from scipy.optimize import root
from enum import Enum
from statistics import newey_west, sample_cov, t_test
from scipy.stats import t

class LinearGMM:
	def _g_t(self, theta, x_t, y_t, z_t):
		if len(x_t.shape) == 1:
			return (y_t - dot(z_t, theta))*x_t
		else:
			return (y_t - dot(z_t, theta))[:, None]*x_t

	def __init__(self, x_t, y_t, z_t):
		self.x_t = x_t
		self.y_t = y_t
		self.z_t = z_t
		self.n_obs = self.z_t.shape[0]
		self.g_t = self._g_t
		try:
			self.n_eq = x_t.shape[1]
		except:
			self.n_eq = 1
		try:
			self.n_param = z_t.shape[1]
		except:
			self.n_param = 1
		self.coef = {}

	def fit(self, lag=0):
		W = identity(self.n_eq)
		S_xz = sample_cov(self.x_t, self.z_t)
		s_xy = sample_cov(self.x_t, self.y_t) 
		if self.n_eq == self.n_param:
			theta_hat = dot(inv(S_xz).T, s_xy) 
			self.coef['theta'] = theta_hat
		elif self.n_eq>self.n_param:
			temp = dot(dot(S_xz.T, inv(W)), S_xz)
			theta_hat = dot(inv(temp).T, temp ) 
		S_hat = newey_west(self.g_t(theta_hat, self.x_t, self.y_t, self.z_t), lag=lag)
		W = inv(S_hat)
		if self.n_eq > self.n_param:
			temp = dot(dot(S_xz.T, inv(W)), S_xz)
			theta_hat = dot(inv(temp), temp)
			self.coef['theta'] = theta_hat
		self.coef['cov'] = inv(dot(dot(S_xz.T, W), S_xz))
		self.coef['t-test'], self.coef['significance'] = t_test(self.coef['theta'], self.coef['cov'], self.n_obs)

class NonLinearGMM:
	def __init__(self, x_t, y_t, z_t, n_param, n_eq, 
				theta=None, g_t_func=None, G_t_func=None):
		""" 
		x_t is exorgenous variable
		y_t is target variable
		z_t is endogenous variable
		n_parma is number of parameter used in the model
		theta  is the parameter of the model. If theta is None, it means the estimates are given. 
		Otherwise, estimation is required
		g_t is a function to calculate the orthogonality condition
		G_t is a function to calculate the Jocobian of g_t
		"""
		self.x_t = x_t
		self.y_t = y_t
		self.z_t = z_t
		self.n_eq = n_eq
		self.n_obs = z_t.shape[0]
		self.n_param = n_param
		self.coef = {'theta': theta}
		self.g_t_func = g_t_func
		self.G_t_func = G_t_func
		if self.G_t_func is not None and self.g_t_func is not None:
			self.FOC = lambda x, W: dot(dot(mean(self.G_t_func(x, self.x_t, self.y_t, self.z_t), axis=0), W),
											mean(self.g_t_func(x, self.x_t, self.y_t, self.z_t), axis=0)) 
		
	def fit(self, lag=0):
		if self.coef['theta'] is None:
			W = identity(self.n_eq)
			theta_hat = root(self.FOC, x0=ones(self.n_param), args=(W)).x
		g_t = self.g_t_func(theta_hat, self.x_t, self.y_t, self.z_t)
		S_hat = newey_west(g_t, lag=lag)
		W = inv(S_hat)
		theta_hat = root(self.FOC, x0=ones(self.n_param), args=(W)).x
		G_T = mean(self.G_t_func(theta_hat, self.x_t, self.y_t, self.z_t), axis=0)
		self.coef['theta'] = theta_hat
		self.coef['cov'] = inv(dot(dot(G_T, W), G_T.T))
		self.coef['t-test'], self.coef['significance'] = t_test(self.coef['theta'], self.coef['cov'], self.n_obs)






