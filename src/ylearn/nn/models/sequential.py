import numpy as np
# import nnfs
# from nnfs.datasets import spiral_data

from ..loss import SoftmaxCategoricalCrossEntropy, CategoricalCrossEntropy, Accuracy
from ..layers import Dense, Dropout, Input
from ..optimizers import Adam
from .model import Model
from ..activation import Softmax

class Sequential(Model):
	
	def __init__(self):
		self.layers = []
		self.built = False
	
	def add(self, layer):
		self.layers.append(layer)
	
	# Check if the layers are valid and prepare them for fit
	def build(self, loss_function = Accuracy(), optimizer = Adam()):
		self.optimizer = optimizer
		self.loss_function = loss_function

		if len(self.layers) < 1:
			return
		
		# Prepare the input layer
		self.input_layer = Input()
		
		no_layers = len(self.layers)
		for i in range(no_layers):
			if i == 0:
				self.layers[i].previous = self.input_layer
				self.layers[i].next = self.layers[i + 1]
			elif i == no_layers - 1:
				self.layers[i].next = self.loss_function
				self.layers[i].previous = self.layers[i-1]
			else:
				self.layers[i].next = self.layers[i+1]
				self.layers[i].previous = self.layers[i-1]
			
			# Check compatibility
			if not self.layers[i].n_input:
				self.layers[i].n_input = self.layers[i-1].n_output
			
			self.layers[i].build() 
				
		# Check if the loss function is CategoricalCrossEntropy 
		# and activation function is softmax
		# if isinstance(self.layers[-1].activation, Softmax) and \
		# 	isinstance(self.loss_function, CategoricalCrossEntropy):
		# 	self.loss_function = SoftmaxCategoricalCrossEntropy()
		# 	self.layers[-1].activation = self.loss_function
		
		self.built = True
	
	def fit(self, X, y, epochs = 1 , batch_size = None, 
		print_every = 1 , checkpoint = False, validation_data = None):
		if not self.built:
			raise Exception("Illegal state, must build first")

		total = len(X) 

		# if the batch_size is not defined and the data is 
		# larger than 128 set the batch size to 128
		if not batch_size and total > 128:
			batch_size = 128

		train_steps = total // batch_size
		# Dividing rounds down. If there are some remaining
		# data but not a full batch, this won't include it
		# Add `1` to include this not full batch
		if train_steps * batch_size < total:
			train_steps += 1


		# Train
		for i in range(epochs):
			
			self.loss_function.new_pass()
			for step in range(train_steps):
				train_X = X[step * batch_size:(step + 1) * batch_size]
				train_y = y[step * batch_size:(step + 1) * batch_size]

				current_loss = 0
				regularaization_loss = 0
				current_total_loss = 0
				
				# forward pass
				self.input_layer.forward(train_X)
				for layer in self.layers:
					layer.forward(layer.previous.output, training = True)
				
				# Get Predictions
				current_output = self.layers[-1].activation#predictions()
				
				current_loss, regularaization_loss = self.loss_function.calculate(current_output, train_y)
				
				# Get Loss
				# for layer in reversed(self.layers):
				# 	regularaization_loss += self.loss_function.regularaization_loss(layer)
				
				# current_total_loss = current_loss + regularaization_loss

				# Do backward steps
				self.loss_function.backward(self.layers[-1].activation.output, train_y)
				for layer in reversed(self.layers):
					layer.backward(layer.next.dinputs)

				# Update weights 
				self.optimizer.pre_update_params()
				for layer in reversed(self.layers):
					if hasattr(layer, 'weights'):
						self.optimizer.update_params(layer)
				self.optimizer.post_update_params()

				# TODO: Validation
				# TODO: Checkpoint and log 
				# TODO: End of on epoch
				# Print a summary
				print(f'step: {i}, ' +
					f'loss: {current_loss:.3f} ' +
					f'lr: {self.optimizer.current_learning_rate}')
				
		
	def predict(self, data):
		self.input_layer.forward(data, training = False)
		for layer in self.layers:
			layer.forward(layer.previous.output, training = False)

		return self.layers[-1].output

	def desc(self):
		for layer in self.layers:
			layer.desc()
		print("Loss: " + self.loss_function.__class__.__name__)