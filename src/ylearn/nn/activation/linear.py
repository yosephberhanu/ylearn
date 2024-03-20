import numpy as np

class Linear:
	# Perform the linear activation function
	def forward(self, inputs):
		# Remember input values for backpropagation
		self.inputs = inputs
		# Return the inputs as is
		self.output = inputs

	# Perform the backward pass
	def backward(self, dvalues):
		# derivative is 1, 1 * dvalues = dvalues - the chain rule
		self.dinputs = dvalues.copy()

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return outputs