import math
import numpy as np
import autograd.numpy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from autograd import elementwise_grad
import autograd.numpy as anp

def sigmoid_one_value(value):
    # print(value.shape)
    
    a = math.exp(-value)
    return 1.0/(1.0+a)
    

def sigmoid_function(value):
    print(type(value))
    value = np.array(value)
    print(type(value))
    a = np.exp(-value)
    return 1.0/ (1.0 + a)

    


grad_sigmoid = elementwise_grad(sigmoid_function)

class logisticregression:
    def __init__(self,regularization="No"):
        self.regularization=regularization
        self.weights = []

    def fit(self,X,y,intercept_addition=True,n_iterations = 100,learning_rate = 0.01,use_jax=False):
        print(X.shape)
        print(y.shape)
        if(intercept_addition==True):
            self.intercept_addition=True
            column = np.ones((X.shape[0],1))
            X = np.hstack((column,X))
        if(self.regularization=="No"):            
            self.weights = np.ones(X.shape[1])
            for i in range(n_iterations):
                ### found the predictions for a step 
                values = np.dot(X,self.weights)
                # print(values)
                
                predicted_vals = sigmoid_function(values)
                error_val = (y-predicted_vals)
                
                # break
                if(use_jax==False):
                    ## calculating likelihood gradient 
                    gradient_displacement = np.dot(X.T,error_val)
                    self.weights += learning_rate*gradient_displacement
                else:
                    def cost_function(weights):
                        weights = np.array(weights)
                        ab = np.dot(X,weights)
                        ab = np.array(ab)
                        ab = sigmoid_function(ab)
                        cost = np.sum(np.abs(y-ab)**2)
                        return cost
                    grad_cost = elementwise_grad(cost_function)
                    gradient_displacement = grad_cost(anp.array(self.weights))
                    gradient_displacement = np.array(gradient_displacement)
                    self.weights += -learning_rate*gradient_displacement

    
    def predict(self,X):
        if(self.intercept_addition==True):
            column = np.ones((X.shape[0],1))
            X = np.hstack((column,X))
        print("prediction started")
        print(self.weights.shape)
        print(X.shape)
        a = np.dot(X,self.weights)
        a[a>=0.5] = 1
        a[a<0.5] = 0
        print(a.shape)
        return a



X,y = load_breast_cancer(return_X_y=True)
sc = StandardScaler().fit(X)
X = sc.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0)
lr = logisticregression(regularization="No")
lr.fit(X_train,y_train,intercept_addition=True,n_iterations=1000,learning_rate=0.01,use_jax=True)

prediction = lr.predict(X_test)
print(y_train.shape)
print((prediction==y_test).sum())