from .dso import DifferentiableSymbolicOperation
from .const import Const
from .sum import Sum
from .mul import Mul
from .sub import Sub
from .div import Div
class Var(DifferentiableSymbolicOperation):
	def __init__(self, name, value):
		self.name = name
		self.value = value
	def compute(self):
		if self.value is None:
			raise ValueError('unassigned variable')
		return self.value
	
	# \frac{\mathrm{d} }{\mathrm{d} x}x = 1
	# \frac{\mathrm{d} }{\mathrm{d} x}k = 0
	def backward(self, var):
		return Const(1) if self == var else Const(0)
	
	def forward(self):
		return self.value, 0
	
	def __repr__(self):
		return f'{self.name}'

	@staticmethod
	def _to_symbolic(x):
		'''
		makes sure that x is a tree node by converting it
		into a constant node if necessary
		'''
		if not isinstance(x, DifferentiableSymbolicOperation):
			return Const(x)
		else:
			return x
	def __add__(self, other):
		return Sum(self, self._to_symbolic(other))

	def __mul__(self, other):
		return Mul(self, self._to_symbolic(other))
	
	def __sub__(self, other):
		return Sub(self, self._to_symbolic(other))
	
	def __floordiv__(self, other):
		return Div(self, self._to_symbolic(other))
	
	def __truediv__(self, other):
		return Div(self, self._to_symbolic(other))
	
	def __neg__(self):
		return Mul(Const(-1), self)