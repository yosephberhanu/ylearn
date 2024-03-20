import numpy as np
from ..loss import Loss
from ..loss import CategoricalCrossEntropy
from ..activation import Softmax
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class SoftmaxCategoricalCrossEntropy(CategoricalCrossEntropy):
	# Creates activation and loss function objects
	def __init__(self):
		self.activation = Softmax()
		self.loss = CategoricalCrossEntropy()
	# Forward pass
	def calculate(self, inputs, y_true):
		# Output layer's activation function
		self.activation.forward(inputs)
		# Set the output
		self.output = self.activation.output
		# Calculate and return loss value
		return self.loss.calculate(self.output, y_true)

	# Backward pass
	def backward(self, dvalues, y_true):
		# Number of samples
		samples = len(dvalues)

		# If labels are one-hot encoded, turn them into discrete values
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)

		# Copy so we can safely modify
		self.dinputs = dvalues.copy()
		
		# Calculate gradient
		self.dinputs[range(samples), y_true] -= 1
		
		# Normalize gradient
		self.dinputs = self.dinputs / samples
	def desc(self):
		print(f"\t SoftmaxCategoricalCrossEntropy")
