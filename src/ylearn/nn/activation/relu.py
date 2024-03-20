import numpy as np

class ReLU:
	# Perform the ReLU activation
	def forward(self, inputs):
		# Remember input values for backpropagation
		self.inputs = inputs
		# Return zero for negative values, otherwise return as is
		self.output = np.maximum(0, inputs)

	# Perform the backward pass
	def backward(self, dvalues):
		# Since we need to modify original variable, let's make a copy of values first
		self.dinputs = dvalues.copy()
		# Zero gradient where input values were negative
		self.dinputs[self.inputs <= 0] = 0
	def desc(self):
		print(f"\t ReLU")

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return outputs