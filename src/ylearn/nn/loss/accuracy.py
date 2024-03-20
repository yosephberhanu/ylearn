import numpy as np
from ..loss import Loss
class Accuracy(Loss):
	def calculate(self, y_pred, y_true):
		y_pred = np.argmax(y_pred, axis = 1)

		# Check if the y_true is one_hot encoded
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis = 1)
		
		# Calculate average accuracy
		return np.mean(y_pred == y_true)
	def backward(self, dvalues, y_true):
		pass