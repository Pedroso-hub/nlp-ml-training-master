import os
import sys
import numpy as np

# Code from: https://www.youtube.com/watch?v=t2ym2a3pb_Y


class Perceptron():

    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Init weights.
        # Try to initialize them using random values. (np.random.random(n_features))
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for i in range(self.n_iter):
            print(f"Epoch: {i}")
            for ind, x_i in enumerate(X):
                """
                    Calculating the output.
                    input: [1, 0]
                    weights: [0.02, 0.01]
                    bias: -0.03

                    input * weights + bias
                    
                    linear_out = 1*0.02 + 0*0.01 + (-0.03)
                    linear_out = 0.02 + 0 - 0.03
                    linear_out = -0.01
                """
                linear_out = np.dot(x_i, self.weights) + self.bias
                """
                    Activation

                    Our threshold is 0, thus, if it is 0 or higher, the activation is 1.
                    On the other hand, if it is smaller than zero, the activation is 0.

                    activation_func(linear_out)
                    activation_func(-0.01) = 0
                """
                y_pred = self.activation_func(linear_out)
                print(f"\tPredicting {x_i}: {linear_out}\n\tAfter activation: {y_pred}")  
                update = self.lr * (y_[ind] - y_pred)
                print(f"\tProposed update: {update}")
                update_w = update * x_i
                self.weights += update_w
                self.bias += update
                print(f"New weights and bias: {self.weights}, {self.bias}\n--\n")
            print(f"Weights: {self.weights} and {self.bias}")


    def predict(self, X):
        linear_out = np.dot(X, self.weights)  + self.bias
        y_pred = self.activation_func(linear_out)

        return y_pred