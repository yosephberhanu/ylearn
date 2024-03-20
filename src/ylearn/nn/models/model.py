import numpy as np
import os
import pickle
import copy

class Model:
	
	def add(self, layer):
		self.layers.append(layer)
	
	# Check if the layers are valid and prepare them for fit
	def build(self):
		pass
	
	def fit(self, X, y):
		pass
	
	def predict(self, data):
		pass

	def save(self):
		pass

	@staticmethod
	def load(path):
		pass