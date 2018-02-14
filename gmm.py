
import numpy as np
from  numpy import  inner, ones, identity
from numpy.linalg import inv
from scipy.optimize import root
import pandas as pd
from enum import Enum
from statistics import newey_west, sample_cov

MODEL_TYPE = Enum('model',  'LINEAR NONLINEAR')

class GMM:
	def __init__(self, x_t, y_t, z_t, n_param, n_eq, theta=None, g_t_func=None, G_t_func=None, model_type=MODEL_TYPE.LINEAR):
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
		self.n_param = n_param
		self.theta_hat = theta
		self.g_t_func = g_t_func
		self.G_t_func = G_t_func
		if self.G_t_func is not None and self.g_t_func is not None:
			self.FOC = lambda x, W: inner(inner(np.mean(self.G_t_func(x, self.x_t, self.y_t, self.z_t), axis=1), W), 
												 np.mean(self.g_t_func(x, self.x_t, self.y_t, self.z_t), axis=0)) 
		self.model_type = model_type
	def train(self, lag=0):
		if self.theta_hat is None:
			W = identity(self.n_eq)
			if self.model_type == MODEL_TYPE.LINEAR:
				S_xz = sample_cov(self.x_t, self.z_t)
				s_xy = sample_cov(self.x_t, self.y_t) 
				if self.n_eq == self.n_param:
					theta_hat = inner(inv(S_xz), s_xy) 
				elif self.n_eq>self.n_param:
					temp = inner(inner(S_xz, inv(W)), S_xz)
					theta_hat = inner(inv(temp), temp ) 
			elif self.model_type==MODEL_TYPE.NONLINEAR:
				theta_hat = root(self.FOC, x0=ones(self.n_param), args=(W)).x
		g_t = self.g_t_func(theta_hat, self.x_t, self.y_t, self.z_t)
		S_hat = newey_west(g_t, lag=lag)
		W = inv(S_hat)

		if self.model_type == MODEL_TYPE.LINEAR:
			if self.n_eq > self.n_param:
				temp = inner(inner(S_xz, inv(W)), S_xz)
				theta_hat = inner(inv(temp), temp)
			return theta_hat, inv(inner(inner(S_xz, W), S_xz))
		elif self.model_type == MODEL_TYPE.NONLINEAR:
			theta_hat = root(self.FOC, x0=ones(self.n_param), args=(W)).x
			G_T = np.mean(self.G_t_func(theta_hat, self.x_t, self.y_t, self.z_t), axis=1)
			return theta_hat, inv(inner(inner(G_T, W), G_T))






