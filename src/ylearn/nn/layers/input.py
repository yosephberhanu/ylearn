import numpy as np
from ..layers import Layer
class Input(Layer):
	# Perform the forward pass
	def forward(self, inputs, training = False):
		self.output = inputs
	