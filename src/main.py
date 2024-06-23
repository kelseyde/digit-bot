import data_loader

from network import Network
from src.cost.cross_entropy_cost import CrossEntropyCost

net = Network([784, 30, 10], cost=CrossEntropyCost)
training_data, validation_data, test_data = data_loader.load_data_wrapper()
net.stochastic_gradient_descent(training_data, 30, 10, 0.5,
                                evaluation_data=test_data,
                                lmbda=5.0,
                                monitor_evaluation_cost=False,
                                monitor_evaluation_accuracy=True,
                                monitor_training_cost=False,
                                monitor_training_accuracy=True)
net.save("../nets/model.json")
