import math
import numpy as np

from jax import grad
def sigmoid_function(x):
  return 1 / (1 + np.exp(-x))

class logisticregression:
    def __init__(self,regularization="No"):
        self.regularization=regularization
        self.weights = []

    def fit(self,X,y,intercept_addition=True,n_iterations = 100,learning_rate = 0.01,use_jax=False):
        if(self.regularization=="No"):            
            if(intercept_addition==True):
                column = np.ones((X.shape[0],1))
                X = np.hstack((column,X))
            self.weights = np.zeros(X.shape[1])
            for i in range(n_iterations):
                ### found the predictions for a step 
                values = np.dot(X,self.weights)
                predicted_vals = sigmoid_function(values)
                error_val = y-predicted_vals
                if(use_jax==False):
                    ## calculating likelihood gradient 
                    gradient_displacement = np.dot(X.T,error_val)
                    self.weights += learning_rate*gradient_displacement
                else:
                    gradient_displacement = grad(sigmoid_function)(values)
                    self.weights += learning_rate*gradient_displacement
        



       
        
        
        
        
        
        
        return 0
    
    def predict(self,X):
        return 0


