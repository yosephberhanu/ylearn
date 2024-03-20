import numpy as np
class Cosine:
	def distance(self, A, B):
		# The distance can then be calculated as :
		# 		1 - Cosine similarity
		# Cosine similarity is given by:
		# 		(A dot B)/(||A|| * ||d||)
		return 1 - np.dot(A, B)/(np.norm(A) * np.norm(B))