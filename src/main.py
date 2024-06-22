import data_loader
from network import Network

net = Network([784, 30, 10])
training_data, validation_data, test_data = data_loader.load_data_wrapper()
net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)