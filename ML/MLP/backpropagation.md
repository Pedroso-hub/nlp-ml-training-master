# Back propagation

### Error Function

E = E(p, y) = 1/2(p - y)^2

### Calculate gradient of error function

Calculate the derivative of the Error function in respect of W (all the weights).

Neural Network

F = F(x, W)
E = E(p, y) = E(F(x, W), y)

Considering a neural network with three layers:

[Alt text](https://www.researchgate.net/profile/Rocco-Langone/publication/45267123/figure/fig2/AS:306016366415878@1449971395841/Feedforward-neural-network-with-a-single-hidden-layer-and-one-output-neuron.png)

We have three input nodes: x1, x2, and x3
Connected to four hidden nodes: h1, h2, and h3
Connected to two output nodes: o1

Each node has its activations according to the value in it. For the input layer, the activation is the x's themselves. But to formalize it, we will use the following representations.

**Input layer activations**: a1,1; a1,2; a1,3

The first number after 'a' indicates that we are talking about the activations from the first layer. While the second number indicates that the node in the layer.

For the hidden layer, we will have a similar representation.

**Hidden layer activations**: a2,1; a2,2; a2,3

**Output layer activations**: a3,1

The weights for our neural network are matrices that connect our layers. Thus, we can represent them as:

**W(1)**: Connects the Input layer with the Hidden layer. Matrix (3,3)
**W(2)**: Connects the Hidden layer with the Output layer. Matrix (3,1)

To calculate the derivatives of the error with respect to the weights, we use the chain rule.

**dE/dW(2)** = dE/da3 * da3/do1 * do1/dW(2)

- dE/da3 = a3 - y

E = 1/2(a3 - y)^2

- da3/do1 = sigmoid'(o1) = sigmoid(o1)(1-sigmoid(o1))

a3 = sigmoid = 1/(1+ e^(-o1))

do1/dW(2) = a2

o1 = a2 * W(2)

**dE/dW(2)** = (a3 - y) * (sigmoid'(o1)) * (a2)

**dE/dW(1)** = dE/da2 * da2/dh1 * dh1/dW(1)

dE/da2 = dE/da3 * da3/do1 * do1/da2

**dE/dW(1)** = (dE/da3 * da3/do1) * do1/da2 * da2/dh1 * dh1/dW(1)

do1/da2 = a2 * W(2) = W(2)

da2/dh1 = sigmoid'(h1) = sigmoid(h1) (1 - sigmoid(h1))

dh1/dW(1) = xW(1) = x

**dE/dW(1)** = (a3 - y) * (sigmoid'(o1)) * W(2) * sigmoid'(h1) * x


# Gradient Descent

Take a step in the opposite direction to the gradient.
The size of the step is the learning rate.