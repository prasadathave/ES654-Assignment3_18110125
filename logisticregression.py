import math
import numpy as np
import autograd.numpy
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from autograd import elementwise_grad
import autograd.numpy as anp
from sklearn.model_selection import KFold

def sigmoid_one_value(value):
    # print(value.shape)
    
    a = math.exp(-value)
    return 1.0/(1.0+a)
    

def sigmoid_function1(value):
   
    a = anp.exp(-value)
    return 1.0/ (1.0 + a)


def sigmoid_function(value):
   
    a = np.exp(-value)
    return 1.0/ (1.0 + a)

    


grad_sigmoid = elementwise_grad(sigmoid_function)

class logisticregression:
    def __init__(self,regularization="No"):
        self.regularization=regularization
        self.weights = []

    def fit(self,X,y,intercept_addition=True,n_iterations = 100,learning_rate = 0.01,use_autograd=False):

        def cost_function_normal(weights):
            # weights = np.array(weights)
            ab = anp.dot(X,weights)
            ab = anp.array(ab)
            ab = sigmoid_function1(ab)
            cost = np.sum(np.abs(y-ab)**2)
            return cost
        def cost_function_L1(weights,lmbda=0.1):
            # weights = np.array(weights)
        
            ab = anp.dot(X,weights)
            ab = anp.array(ab)
            ab = sigmoid_function1(ab)
            cost = np.sum(np.abs(y-ab)**2)
            cost += lmbda*(np.sum(np.abs(weights)))
            return cost
        
        def cost_function_L2(weights,lmbda=0.1):
            # weights = np.array(weights)
        
            ab = anp.dot(X,weights)
            ab = anp.array(ab)
            ab = sigmoid_function1(ab)
            cost = np.sum(np.abs(y-ab)**2)
            cost += lmbda*anp.dot(weights.T,weights)
            return cost
        


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
                if(use_autograd==False):
                    ## calculating likelihood gradient 
                    gradient_displacement = np.dot(X.T,error_val)
                    self.weights += learning_rate*gradient_displacement
                else:
                    grad_cost = elementwise_grad(cost_function_normal)
                    gradient_displacement = grad_cost(anp.array(self.weights))
                    gradient_displacement = np.array(gradient_displacement)
                    self.weights += -learning_rate*gradient_displacement
        elif(self.regularization=="L1"):
            self.weights = np.ones(X.shape[1])
            for i in range(n_iterations):
                ### found the predictions for a step 
                values = np.dot(X,self.weights)
                # print(values)
                
                predicted_vals = sigmoid_function(values)
                error_val = (y-predicted_vals)
                

                grad_cost_L1 = elementwise_grad(cost_function_L1)
                gradient_displacement = grad_cost_L1(anp.array(self.weights))
                gradient_displacement = np.array(gradient_displacement)
                self.weights += -learning_rate*gradient_displacement

        elif(self.regularization=="L2"):
            self.weights = np.ones(X.shape[1])
            for i in range(n_iterations):
                ### found the predictions for a step 
                values = np.dot(X,self.weights)
                # print(values)
                
                predicted_vals = sigmoid_function(values)
                error_val = (y-predicted_vals)
                
                grad_cost_L2 = elementwise_grad(cost_function_L2)
                gradient_displacement = grad_cost_L2(anp.array(self.weights))
                gradient_displacement = np.array(gradient_displacement)
                self.weights += -learning_rate*gradient_displacement
            


    
    def predict(self,X):
        if(self.intercept_addition==True):
            column = np.ones((X.shape[0],1))
            X = np.hstack((column,X))
        # print("prediction started")
        # print(self.weights.shape)
        # print(X.shape)
        a = np.dot(X,self.weights)
        a[a>=0.5] = 1
        a[a<0.5] = 0
        # print(a.shape)
        return a


#### importing the breast cancer dataset and transforming it
X,y = load_breast_cancer(return_X_y=True)
sc = StandardScaler().fit(X)
X = sc.transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0)
# lr = logisticregression(regularization="No")
# lr.fit(X_train,y_train,intercept_addition=True,n_iterations=1000,learning_rate=0.01,use_autograd=False)

# prediction = lr.predict(X_test)

# print(y_train.shape)
# print((prediction==y_test).sum())

kf = KFold(n_splits=3)
acc =0
for train, test in kf.split(X):
    X_train,y_train,X_test,y_test = X[train],y[train],X[test],y[test]
    lr = logisticregression(regularization="L2")
    lr.fit(X_train,y_train,intercept_addition=True,n_iterations=100,learning_rate=0.01,use_autograd=True)
    prediction = lr.predict(X_test)
    acc += ((prediction==y_test).sum()/(y_test.shape[0]))*100
    
print(acc/3)

# without autograd = 95.95841455490579
# with autograd = 97.54107490949598
