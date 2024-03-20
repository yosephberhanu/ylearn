import numpy as np
from ..loss import Loss
class MeanSquaredError(Loss):
	def calculate(self, y_pred, y_true):
		# Check if the y_true is one_hot encoded
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis = 1)
		
		# Calculate average mean squared error
		return np.mean((y_pred - y_true)**2)
	def backward(self, dvalues, y_true):
		pass