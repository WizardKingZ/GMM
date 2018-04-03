from abc import ABCMeta, abstractmethod

class BaseRegressionEstimator:
	"""
	Base Regression Class
	"""
	__metaclass__ = ABCMeta

	@abstractmethod
	def fit(self, **kwargs):
		raise NotImplementedError("Should implement fit()")

	@abstractmethod
	def predict(self, *args):
		raise NotImplementedError("Should implement predict()")

	def _check_if_fitted(self):
		return self._coef is not None

	def get(self, attr):
		if self._check_if_fitted():
			if attr == '_coef':
				return self._coef
			elif attr == '_res':
				return self._res
			elif attr == '_var':
				return self._var
		else:
			print "Model is not fitted"	
