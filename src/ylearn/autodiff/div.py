from .dso import DifferentiableSymbolicOperation
from .mul import Mul
from .sub import Sub

class Div(DifferentiableSymbolicOperation):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	# Quotient rule of differentiation
	# \frac{\mathrm{d} }{\mathrm{d} x} \frac{f(x)}{g(x)} = 
	#			\frac{ 
	#					 g(x) * \frac{\mathrm{d} }{\mathrm{d} x}f(x) 
	#									-
	#					 f(x) * \frac{\mathrm{d} }{\mathrm{d} x}g(x) 
	#				 }
	#			{g(x) * g(x)}
	
	def backward(self, var):
		# return Div(
		# 			Sub( 
		# 				Mul(self.x.backward(var), self.y.compute()),
		# 				Mul(self.x.compute(), self.y.backward(var))
		# 			),
		# 			Mul(self.y.compute(), self.y.compute())
		# 		)
		return Div(Sub(Mul(self.y, self.x.backward(var)), Mul(self.x, self.y.backward(var))), Mul(self.y, self.y))  # This line assumes 'var' has a 'grad' attribute to accumulate gradients
	def forward(self):
		# Compute the forward pass for both x and y
		x_value, x_derivative = self.x.forward()
		y_value, y_derivative = self.y.forward()
		
		# Compute the value of the division
		value = x_value / y_value
		
		# Apply the quotient rule to compute the derivative
		derivative = (x_derivative * y_value - x_value * y_derivative) / (y_value * y_value)

		return value, derivative
	def compute(self):
		return self.x.compute() / self.y.compute()
	
	def __repr__(self):
		return f'{self.x} / {self.y}'