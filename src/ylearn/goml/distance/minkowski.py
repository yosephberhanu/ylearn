import numpy as np


class Minkowski:
    def __init__(self, p=0):
        self.p = p

    def distance(self, A, B):
        return np.sum(np.abs(A-B) ** p, axis=1)**(1/p)
