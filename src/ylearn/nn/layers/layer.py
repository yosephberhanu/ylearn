import numpy as np
from ..activation import relu
class Layer:
	def predictions(self):
		return self.activation.predictions(self.output)