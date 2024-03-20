from .dso import DifferentiableSymbolicOperation
from .sum import Sum
class Mul(DifferentiableSymbolicOperation):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	# Product rule of differentiation
	# \frac{\mathrm{d} }{\mathrm{d} x}(f(x)*g(x)) = 
	#			g(x) * \frac{\mathrm{d} }{\mathrm{d} x}f(x) 
	#					+ 
	#			f(x) * \frac{\mathrm{d} }{\mathrm{d} x}g(x) 
	def backward(self, var):
		return Sum(
				Mul (self.x.backward(var), self.y),
				Mul (self.x, self.y.backward(var))
			)
	def forward(self):
		# Compute the forward pass for both x and y
		x_value, x_derivative = self.x.forward()
		y_value, y_derivative = self.y.forward()
		
		# Compute the value of the division
		value = x_value * y_value
		
		# Apply the product rule to compute the derivative
		derivative = y_value*x_derivative - x_value * y_derivative

		return value, derivative
	def compute(self):
		return self.x.compute() * self.y.compute()
	
	def __repr__(self):
		return f'{self.x} * {self.y}'