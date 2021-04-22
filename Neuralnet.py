# import jax.numpy as np

# class neuralnetwork:
    
#     def __init__(self,layer_arr,activation_function_arr=None,loss_function="mse"):
#         ### In this function one should give number of nodes in all layers in array ####
#         ### In activation function which should be used mentioned for each layer from first to last ####
#         if(activation_function_arr!=None):
#             assert(len(activation_function_arr)==len(layer_arr)-1)
#         else:
#             activation_function_arr = ["relu"]*len(activation_function_arr)
#         self.layer_arr = layer_arr
#         self.activation_function_arr = activation_function_arr
#         weights = []

#         for i in range(1,len(layer_arr)):
#             l1 = np.ones([layer_arr[i],layer_arr[i-1]])
#             weights.append(l1)

#         self.weights = weights



import autograd.numpy as np
from jax import grad,jit,vmap
from autograd import elementwise_grad,grad
# inherit from base class Layer
class FullyConnectedLayer:
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size,input1=None,output1=None):
        self.input = input1
        self.output = output1
        self.weights = np.ones([input_size,output_size]) - 0.5
        self.bias = np.ones([1,output_size]) - 0.5

    # returns output for a given input
    def propagate_forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def propagate_backward(self, output_error, learning_rate):
        # print(self.weights)
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error





class ActivationLayer:
    def __init__(self, activation,input1=None,output1=None):
        self.input = input1
        self.output = output1
        self.activation = activation
        self.activation_prime = elementwise_grad(activation)

    # returns the activated input
    def propagate_forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def propagate_backward(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error




# activation function and its derivative
def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def relu(x):
    return 0.1*(abs(x)+x)/2

def mse(y_true,y_pred):
    return np.mean(np.power(y_pred-y_true, 2))


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add_layer(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss):
        self.loss = loss
        self.loss_prime = elementwise_grad(loss)

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        predictions = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for i in range(len(self.layers)):
                output = self.layers[i].propagate_forward(output)
            predictions.append(output)

        return predictions

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.propagate_forward(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = -self.loss_prime(y_train[j], output)
                # print(mse_prime(y_train[j],output))
                # print(mse_prime1(y_train[j],output))
                for p in range(len(self.layers)-1,-1,-1):
                    error = self.layers[p].propagate_backward(error, learning_rate)
                    

            # calculate average error on all samples
            err /= samples
            
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))



x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add_layer(FullyConnectedLayer(2, 3))
net.add_layer(ActivationLayer(relu))
net.add_layer(FullyConnectedLayer(3, 1))
net.add_layer(ActivationLayer(relu))


net.use(mse)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)


out = net.predict(x_train)
