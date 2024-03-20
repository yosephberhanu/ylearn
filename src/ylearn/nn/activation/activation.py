import numpy as np
from .softmax import Softmax
from .relu import ReLU
from .linear import Linear
from .sigmoid import Sigmoid
class Activation(object):
	__activations = { "softmax":Softmax(),
		"relu":ReLU(),
		"linear":Linear(),
		"sigmoid": Sigmoid()
		}
	def __new__(self,name):
		return self.__activations[name]