from keras.utils import np_utils
import autograd.numpy as np
from jax import grad,jit,vmap
from autograd import elementwise_grad,grad
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def relu(x):
    return 0.1*(abs(x)+x)/2

relu_prime = elementwise_grad(relu)
sigmoid_prime = elementwise_grad(sigmoid)




import numpy as np 
from autograd import numpy as np, elementwise_grad
# activation function and its derivative
def tanh(x):
    
    return 1/(1+np.exp(-x))

def tanh_prime(x):
    return 1-np.tanh(x)**2

def mse(y_pred, y_true):
   
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

class FullyConnectedLayer():
    def __init__(self,inputNeurons,outputNeurons):
        self.input = None 
        self.output = None
        self.weigtsMatrix = np.random.rand(inputNeurons, outputNeurons) - 0.5
        self.biasMatrix = np.random.rand(1, outputNeurons) - 0.5 
    
    def forwardPass(self, inputData):
        self.input = inputData 
        self.output = np.dot(self.input,self.weigtsMatrix)+self.biasMatrix
        return self.output

    def backwardPass(self, output_error, learningRate):
        inputError = np.dot(output_error,self.weigtsMatrix.T) 
        weightsError = np.dot(self.input.T,output_error) 

        self.weigtsMatrix -= learningRate*weightsError 
        self.biasMatrix -= learningRate*output_error
        return inputError

class ActivationLayer():
    def __init__(self, activation, activationPrime):
        self.input = None 
        self.output = None
        self.activation = activation 
        self.activationPrime = activationPrime 
    
    def forwardPass(self, inputData):
        self.input = inputData 
        self.output = self.activation(self.input)
        return self.output 
    
    def backwardPass(self, output_error, learningRate):
        #autograd
        agrad = elementwise_grad(tanh)
        return agrad(self.input)*output_error #localgradient*upstream gradient 

    





from autograd import numpy as np, elementwise_grad
class NeuralNetwork():

    def __init__(self):
        self.listOfLayers = []
        self.loss = None
        self.lossPrime = None

    def add(self, layer):
        self.listOfLayers.append(layer)
    
    def use(self, loss, loss_prime):
        self.loss = loss
        self.lossPrime = loss_prime
    
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.listOfLayers:
                output = layer.forwardPass(output)
            result.append(output)
        return result 
    
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.listOfLayers:
                    output = layer.forwardPass(output)

                # compute loss (for display purpose only)
                err += self.loss(output, y_train[j])
                # backward propagation
                agrad = elementwise_grad(self.loss)
                error = agrad(output,y_train[j]) #output = z
                for layer in reversed(self.listOfLayers):
                    error = layer.backwardPass(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))



digits = load_digits()
data = digits.images
n = len(data)

X = data.reshape((n,-1))

y = digits.target


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train = np.array(X_train,dtype=int)
y_train = np.array(y_train,dtype=int)
X_train =X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
X_train = X_train.astype('float32')
y_train = y_train.reshape((y_train.shape[0],1,1))
y_train = np_utils.to_categorical(y_train)
# print(X_train.shape)
# print(y_train.shape)
# print(y_train[0])
net = NeuralNetwork()
net.add(FullyConnectedLayer(64, 32))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(relu, relu_prime))
net.add(FullyConnectedLayer(32, 16))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FullyConnectedLayer(16, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(relu, relu_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
net.fit(X_train,y_train,epochs=100,learning_rate=0.6)
out = net.predict(X_train)


print("predicted values : ")
Y = []
for i in out:
    Y.append(np.argmax(i))
# print(out)
out = np.argmax(out,axis=1)

yhat = []
for i in y_train:
    yhat.append(np.argmax(i))

# print((yhat==Y).sum()/y_test.shape[0])
yhat = np.array(yhat)
# print(yhat.shape)
# print(y_test.shape)
print(y_train.shape)
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    # assert(y_hat.size == y.size)
    totalTruth = 0
    for i in range(len(y_hat)):
        if(y_hat[i]==y[i]):
            totalTruth+=1 
    return (totalTruth/len(y_hat))

print(accuracy(yhat,Y))


# epoch 100/100   error=0.016703
# predicted values : 
# (1437, 1, 10)
# 0.9909533750869868