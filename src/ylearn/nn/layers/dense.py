import numpy as np
from .layer import Layer
from ..activation import ReLU
class Dense(Layer):
	
	def __init__(self, n_output, n_input = None, l1 = 0, l2 = 0, activation = ReLU(), trainable = True):
		self.n_input = n_input
		self.n_output = n_output
		self.l1 = l1
		self.l2 = l2
		self.trainable = trainable
		self.activation = activation
	
	def build(self, factor = 0.10):
		# Initialize the weight matrix to random values of 
		# the dimension n_inputs x n_neurons
		# Initialize the bias vector with n_neurons dimensions
		if not self.n_input:
			raise Exception("Layers not compatible")
		
		# Initialize weights and biases
		self.weights = factor * np.random.randn(self.n_input, self.n_output)
		self.bias = np.zeros((1, self.n_output))

	# Perform the forward pass
	def forward(self, inputs, training = True):
		# Remember input values for backpropagation
		self.inputs = inputs
		self.output = np.dot(inputs, self.weights) + self.bias
		self.activation.forward(self.output)
	
	# Perform the backward pass
	def backward(self, dvalues):
		# Perform backward pass on the activation function
		self.activation.backward(dvalues)
		self.dvalues = self.activation.dinputs
		
		# Gradients on parameters
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
		
		# L1 Regularization
		if self.l1 > 0:
			# L1 on weights
			dL1 = np.ones_like(self.weights)
			dL1[self.weights < 0] = -1
			self.dweights += self.l1 * dL1
			
			# L1 on bias
			dL1 = np.ones_like(self.bias)
			dL1[self.bias < 0] = -1
			self.dbiases += self.l1 * dL1
		
		# L2 Regularization
		if self.l2 > 0:
			# L2 on weights
			self.dweights += 2 * self.l2 * self.weights
			# L2 on bias
			self.dbiases += 2 * self.l2 * self.bias
		
		# Gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)

	def desc(self):
		print(f"Dense - ({self.n_input}x{self.n_output}), weights", self.weights.shape,", bias", self.bias.shape)
		self.activation.desc()