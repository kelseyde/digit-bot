import numpy as np

from src.utils import sigmoid_prime


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output `a` and desired output `y`."""
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)
