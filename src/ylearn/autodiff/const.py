from .dso import DifferentiableSymbolicOperation
class Const(DifferentiableSymbolicOperation):
	def __init__(self, value):
		self.value = value

	def compute(self):
		return self.value

	# \frac{\mathrm{d} }{\mathrm{d} x}k = 0
	def backward(self, var):
		# Construct new node, does not perform the actual operations.
		# In this way, we will have a new computational graph representing 
		# the derivative with respect to the variable.
		return Const(0)
	
	def forward(self):
		return self.value, Const(0)
	
	def __repr__(self):
		return str(self.value)