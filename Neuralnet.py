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

relu_derivative = elementwise_grad(relu)
sigmoid_derivative = elementwise_grad(sigmoid)




import numpy as np 
from autograd import numpy as np, elementwise_grad
# activation function and its derivative
def tanh(x):
    
    return 1/(1+np.exp(-x))

def tanh_derivative(x):
    return 1-np.tanh(x)**2

def mse(y_pred, y_true):
   
    return np.mean(np.power(y_true-y_pred, 2))

def mse_derivative(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

class fullconnectedlayer():
    def __init__(self,input_number,output_number):
        self.input = None 
        self.output = None
        self.matrix_of_weights = np.random.rand(input_number, output_number) - 0.5
        self.matrix_of_biases = np.random.rand(1, output_number) - 0.5 
    
    def propogate_forward(self, inputData):
        self.input = inputData 
        self.output = np.dot(self.input,self.matrix_of_weights)+self.matrix_of_biases
        return self.output

    def propogate_backward(self, dout, learning_rate):
        dinput = np.dot(dout,self.matrix_of_weights.T) 
        dW = np.dot(self.input.T,dout) 

        self.matrix_of_weights -= learning_rate*dW 
        self.matrix_of_biases -= learning_rate*dout
        return dinput

class Alayer():
    def __init__(self, activation):
        self.input = None 
        self.output = None
        self.activation = activation 

    def propogate_forward(self, inputData):
        self.input = inputData 
        self.output = self.activation(self.input)
        return self.output 
    
    def propogate_backward(self, dout, learning_rate):
        #autograd
        derivative_of_function = elementwise_grad(self.activation)
        return derivative_of_function(self.input)*dout #localgradient*upstream gradient 

    





from autograd import numpy as np, elementwise_grad
class neuralnet():

    def __init__(self):
        self.layer_array = []
        self.loss = None


    def add_layer(self, layer):
        self.layer_array.append(layer)
    
    def select_loss_function(self, loss):
        self.loss = loss
        
    
    def predict(self, input_data):
        # sample dimension first
        number_of_samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(number_of_samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layer_array:
                output = layer.propogate_forward(output)
            result.append(output)
        return result 
    
    def fit(self, x_train, y_train, number_of_epochs, learning_rate):
        # sample dimension first
        number_of_samples = len(x_train)
        # training loop
        for i in range(number_of_epochs):
            err = 0
            for j in range(number_of_samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layer_array:
                    output = layer.propogate_forward(output)

                
                err += self.loss(output, y_train[j])
                # backward propagation
                derivative_of_loss_function = elementwise_grad(self.loss)
                error = derivative_of_loss_function(output,y_train[j]) #output = z
                for layer in reversed(self.layer_array):
                    error = layer.propogate_backward(error, learning_rate)

            # calculate average error on all number_of_samples
            err /= number_of_samples
            print('epoch number %d/%d   loss=%f' % (i+1, number_of_epochs, err))


##################### training using digits dataset #########################


# digits = load_digits()
# data = digits.images
# n = len(data)

# X = data.reshape((n,-1))

# y = digits.target


# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# X_train = np.array(X_train,dtype=int)
# y_train = np.array(y_train,dtype=int)
# X_train =X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
# X_train = X_train.astype('float32')
# y_train = y_train.reshape((y_train.shape[0],1,1))
# y_train = np_utils.to_categorical(y_train)



# net = neuralnet()
# net.add_layer(fullconnectedlayer(64, 50))               
# net.add_layer(Alayer(relu))

# net.add_layer(fullconnectedlayer(50, 32))                   
# net.add_layer(Alayer(relu))


# net.add_layer(fullconnectedlayer(32, 16))                   
# net.add_layer(Alayer(sigmoid))

# net.add_layer(fullconnectedlayer(16, 10))                    
# net.add_layer(Alayer(relu))


# net.select_loss_function(mse)
# net.fit(X_train,y_train,number_of_epochs=10,learning_rate=0.9)
# out = net.predict(X_train)






##############################boston dataset###############################################

X,y = load_boston(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train = np.array(X_train,dtype=int)
y_train = np.array(y_train,dtype=int)
X_train =X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
X_train = X_train.astype('float32')
y_train = y_train.reshape((y_train.shape[0],1,1))
y_train = np_utils.to_categorical(y_train)

# print(y_train)


net = neuralnet()
net.add_layer(fullconnectedlayer(13, 10))               
net.add_layer(Alayer(tanh))

net.add_layer(fullconnectedlayer(10, 8))                   
net.add_layer(Alayer(relu))


net.add_layer(fullconnectedlayer(8, 1))                   
net.add_layer(Alayer(sigmoid))


net.select_loss_function(mse)
net.fit(X_train,y_train,number_of_epochs=10,learning_rate=0.9)
out = net.predict(X_train)

#######################################################################




print(y_train)
print(out)
# print("predicted values : ")
# Y = []
# for i in out:
#     Y.append(np.argmax(i))
# # print(out)
# out = np.argmax(out,axis=1)

# yhat = []
# for i in y_train:
#     yhat.append(np.argmax(i))

# # print((yhat==Y).sum()/y_test.shape[0])
# yhat = np.array(yhat)
# # print(yhat.shape)
# # print(y_test.shape)

# def accuracy(y_hat, y):
#     num = 0
#     for i in range(len(y_hat)):
#         if(y_hat[i]==y[i]):
#             num+=1 
#     return (num/len(y_hat))

# print(accuracy(yhat,Y))


# epoch 100/100   error=0.016703
# predicted values : 
# (1437, 1, 10)
# 0.9909533750869868