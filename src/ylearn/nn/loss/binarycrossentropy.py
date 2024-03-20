import numpy as np
from ..loss import Loss

class BinaryCrossEntropy(Loss):
	# Perform the forward pass
	def calculate(self, y_pred, y_true):
		# Clip the predicted values in y_pred to be in the range 1e-7, 1 - 1e-7
		# This ensures that our predictions are never zero which addresses the 
		# issue of log(0) not being defined
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		
		sample_losses = - (y_true * np.log(y_pred_clipped) \
						  + ( 1 - y_true) * np.log( 1 - y_pred_clipped))
		sample_losses = np.mean(sample_losses, axis =- 1 )

		return sample_losses
	
	def backward(self, dvalues, y_true):
		# Number of samples
		samples = len(dvalues)
		
		clipped_dvalues = np.clip(dvalues, 1e-7 , 1 - 1e-7 )
		
		outputs = len (clipped_dvalues[ 0 ])

		# Calculate gradient
		self.dinputs = -(y_true / clipped_dvalues \
						  - ( 1 - y_true) / ( 1 - clipped_dvalues)) / outputs
		# Normalize gradient
		self.dinputs = self.dinputs / samples
