from numpy import inner
from statistics import auto_cov

def delta_decorator(func):
	def func_wrapper(data):
		return inner(inner(func(data), auto_cov(data, lag=0)), func(data))
	return func_wrapper