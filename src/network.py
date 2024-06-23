import sys
import random
import json

import numpy as np

from src.cost.cross_entropy_cost import CrossEntropyCost
from src.utils import sigmoid, sigmoid_prime, vectorized_result


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Initialize a neural network with layers of sizes specified in the sizes list, and random weights and biases.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.biases = [np.random.randn(y, 1)
                       for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, input_data):
        """Return the output of the network for a given input."""
        output_data = input_data
        for b, w in zip(self.biases, self.weights):
            output_data = sigmoid(np.dot(w, output_data) + b)
        return output_data

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate,
                                    lmbda=0.0,
                                    evaluation_data=None,
                                    monitor_evaluation_cost=False,
                                    monitor_evaluation_accuracy=False,
                                    monitor_training_cost=False,
                                    monitor_training_accuracy=False):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        """
        n_training_data = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for epoch in range(epochs):
            # For each epoch, shuffle the training data
            random.shuffle(training_data)
            # ... and then split it into mini-batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n_training_data, mini_batch_size)]

            # Then, for each mini-batch, update the weights and biases via a single iteration of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, lmbda, n_training_data)

            print(f"Epoch {epoch} complete")

            # Optionally print some statistics on the network's cost/accuracy, on both the training and evaluation data
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"- Training data cost: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"- Training data accuracy: {accuracy} ({accuracy/len(training_data):.2%})")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"- Evaluation data cost: {cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"- Evaluation data accuracy: {accuracy} ({accuracy/len(evaluation_data):.2%})")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, learning_rate, lmbda, n):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation to a mini-batch.
        """

        # Initialize the gradients for the biases and weights, starting as all zeros.
        gradients_b = [np.zeros(b.shape) for b in self.biases]
        gradients_w = [np.zeros(w.shape) for w in self.weights]

        for input_data, expected_output_data in mini_batch:
            # For each training example in the mini-batch, calculate the gradient deltas for the biases and weights
            delta_gradients_b, delta_gradients_w = self.backprop(input_data, expected_output_data)
            # Add the gradients for this training example to the running total
            gradients_b = [gradient + delta
                           for gradient, delta in zip(gradients_b, delta_gradients_b)]
            gradients_w = [gradient + delta
                           for gradient, delta in zip(gradients_w, delta_gradients_w)]

        # Finally, update the weights and biases using the gradients calculated for the mini-batch
        self.weights = [(1 - learning_rate * (lmbda / n)) * weight - (learning_rate / len(mini_batch)) * gradient
                        for weight, gradient in zip(self.weights, gradients_w)]
        self.biases = [bias - (learning_rate / len(mini_batch)) * gradient
                       for bias, gradient in zip(self.biases, gradients_b)]

    def backprop(self, input_data, expected_output_data):
        """
        Calculate the gradient of the cost function with respect to the biases and weights using backpropagation.
        """
        # Initialize the gradients for the biases and weights, starting as all zeros.
        gradients_b = [np.zeros(b.shape) for b in self.biases]
        gradients_w = [np.zeros(w.shape) for w in self.weights]

        # Since the input layer does not perform any computations, its activation is simply the input data
        activation = input_data

        # Initialize a list to store the activations of each layer, starting with the input data
        activations = [input_data]

        # Initialize a list to store the weighted inputs of each layer
        weighted_inputs = []

        # For each layer, calculate the weighted input and the activation
        for bias, weights in zip(self.biases, self.weights):
            # The weighted input is the dot product of the weights & the activation of the previous layer, plus the bias
            weighted_input = np.dot(weights, activation) + bias
            weighted_inputs.append(weighted_input)
            # The activation is the result of applying the sigmoid function to the weighted input
            activation = sigmoid(weighted_input)
            activations.append(activation)

        # Calculate the error delta between the activation of the output layer and the expected output.
        # delta = self.cost_derivative(activations[-1], expected_output_data) * sigmoid_prime(weighted_inputs[-1])

        delta = self.cost.delta(weighted_inputs[-1], activations[-1], expected_output_data)

        # The gradient of the cost function with respect to the biases in the output layer is simply the error delta
        gradients_b[-1] = delta

        # The gradient of the cost function with respect to the weights in the output layer is the dot product of the
        # error delta and the activation of the previous layer
        gradients_w[-1] = np.dot(delta, activations[-2].transpose())

        # Back-propagate the error delta to the previous layers by looping backward through the layers of the network,
        # starting from the second-to-last layer and ending at the first hidden layer
        for layer in range(2, self.num_layers):
            # Retrieve the weighted inputs for the current layer
            weighted_input = weighted_inputs[-layer]
            # Calculate the sigmoid prime of the weighted input
            sp = sigmoid_prime(weighted_input)
            # Calculate the error delta for the current layer by back-propagating the error delta from the next layer
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            # The gradient of the cost function with respect to the biases in the current layer is the error delta
            gradients_b[-layer] = delta
            # The gradient of the cost function with respect to the weights in the current layer is the dot product of
            # the error delta and the activation of the previous layer
            gradients_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return gradients_b, gradients_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(input_data)), expected_output_data)
                        for (input_data, expected_output_data) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, actual_outputs, expected_outputs):
        """
        Computes the derivative of the cost function with respect to the output activations.
        Note: actual_outputs and expected_outputs are numpy arrays, so the subtract operation is element-wise.
        """
        return actual_outputs - expected_outputs

    def total_cost(self, data, lmbda, convert=False):
        """
        Return the total cost for the data set `data`.

        The flag `convert` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is the validation or test data.
        """
        cost = 0.0
        for input_data, expected_output_data in data:
            a = self.feedforward(input_data)
            if convert:
                expected_output_data = vectorized_result(expected_output_data)
            cost += self.cost.fn(a, expected_output_data) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def accuracy(self, data, convert=False):
        """
        Return the number of inputs in `data` for which the neural network outputs the correct result. The network's
        output is assumed to be the index of whichever neuron in the final layer has the highest activation.

        The flag `convert` should be set to False if the data set is validation or test data (the usual case), and to
        True if the data set is the training data.
        """
        if convert:
            results = [(np.argmax(self.feedforward(input_data)), np.argmax(expected_output_data))
                       for (input_data, expected_output_data) in data]
        else:
            results = [(np.argmax(self.feedforward(input_data)), expected_output_data)
                       for (input_data, expected_output_data) in data]
        return sum(int(x == y) for (x, y) in results)

    def save(self, filename):
        """Save the neural network to the file `filename`."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    """Load a neural network from the file `filename`. Returns an instance of Network."""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net