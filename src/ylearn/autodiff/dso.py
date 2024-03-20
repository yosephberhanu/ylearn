from abc import ABC, abstractmethod
class DifferentiableSymbolicOperation(ABC):
	@abstractmethod
	def backward(self, var):
		pass
	@abstractmethod
	def compute(self):
		pass