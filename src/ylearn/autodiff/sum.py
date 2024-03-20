from .dso import DifferentiableSymbolicOperation
class Sum(DifferentiableSymbolicOperation):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	# Addition operation's backward step passes on the value as is
	def backward(self, var):
		return Sum(self.x.backward(var), self.y.backward(var))
	
	def forward(self):
		# Compute the forward pass for both x and y
		x_value, x_derivative = self.x.forward()
		y_value, y_derivative = self.y.forward()
		
		# Compute the value of the division
		value = x_value + y_value
		
		derivative = x_derivative + y_derivative

		return value, derivative
	def compute(self):
		return self.x.compute() + self.y.compute()
	
	def __repr__(self):
		return f'{self.x} + {self.y}'