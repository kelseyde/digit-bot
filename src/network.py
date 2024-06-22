import numpy as np
import random


class Network(object):

    def __init__(self, sizes):
        """
        Initialize a neural network with layers of sizes specified in the sizes list, and random weights and biases.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, input_data):
        """Return the output of the network for a given input."""
        output_data = input_data
        for b, w in zip(self.biases, self.weights):
            output_data = self.sigmoid(np.dot(w, output_data) + b)
        return output_data

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n_training_data = len(training_data)
        n_test_data = len(test_data) if test_data else 0
        for epoch in range(epochs):
            # For each epoch, shuffle the training data
            random.shuffle(training_data)
            # ... and then split it into mini-batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n_training_data, mini_batch_size)]
            # Then, for each mini-batch, update the weights and biases via a single iteration of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {n_test_data}")
            else:
                print(f"Epoch {epoch} complete")
                if epoch == epochs - 1:
                    print(f"Final weights: {self.weights}")
                    print(f"Final biases: {self.biases}")

    def update_mini_batch(self, mini_batch, learning_rate):
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
        self.weights = [weight - (learning_rate / len(mini_batch)) * gradient
                        for weight, gradient in zip(self.weights, gradients_w)]
        self.biases = [bias - (learning_rate / len(mini_batch)) * gradient
                       for bias, gradient in zip(self.biases, gradients_b)]

    def backprop(self, input_data, expected_output_data):

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
            activation = self.sigmoid(weighted_input)
            activations.append(activation)

        # Calculate the error delta between the activation of the output layer and the expected output.
        delta = self.cost_derivative(activations[-1], expected_output_data) * self.sigmoid_prime(weighted_inputs[-1])

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
            sp = self.sigmoid_prime(weighted_input)
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
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, actual_outputs, expected_outputs):
        """
        Computes the derivative of the cost function with respect to the output activations.
        Note: actual_outputs and expected_outputs are numpy arrays, so the subtract operation is element-wise.
        """
        return actual_outputs - expected_outputs

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))
