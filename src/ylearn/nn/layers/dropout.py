import numpy as np
from .layer import Layer

class Dropout(Layer):
	
	def __init__(self, rate = 0 , trainable = True):
		self.rate = 1 - rate
		self.n_input = None	
		self.trainable = trainable
	
	def build(self):
		self.n_output = self.n_input
	
	# Perform the forward pass
	def forward(self, inputs, training = True):
		# Remember input values for backpropagation
		self.inputs = inputs

		if not training:
			self.output = inputs.copy()
		else:
			self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) / self.rate
			self.output = inputs * self.binary_mask
	
	# Perform the backward pass
	def backward(self, dvalues):
		# Gradients on parameters
		self.dinputs = dvalues * self.binary_mask

	def desc(self):
		print(f"Dropout - {self.rate}({self.n_input}x{self.n_output})")