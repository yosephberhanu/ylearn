import numpy as np

class Softmax:
	# Perform the SoftMax activation
	def forward(self, inputs):
		# Remember input values for backpropagation
		self.inputs = inputs
		# Exponentiate, but first subtract the maximum value in each	
		# row to avoid the exponential overflow
		exp = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
		# Normalize to bring the output in the range of 0-1	
		# This can be interpreted as the confidence level on the classification
		self.output = exp / np.sum(exp, axis = 1, keepdims = True)
	# Perform the backward pass
	def backward(self, dvalues):
		# Create uninitialized array
		self.dinputs = np.empty_like(dvalues)

		# Zero gradient where input values were negative
		self.dinputs[self.inputs <= 0] = 0

		# Enumerate outputs and gradients
		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
			# Flatten output array
			single_output = single_output.reshape(-1, 1)
			# Calculate Jacobian matrix of the output
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

			# Calculate sample-wise gradient and add it to the array of sample gradients
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
	def predictions(self, outputs):
		return np.argmax(outputs, axis=1)

	def desc(self):
		print(f"\t Softmax")