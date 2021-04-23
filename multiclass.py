import math
import numpy as np
import autograd.numpy
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from autograd import elementwise_grad
import autograd.numpy as anp
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from logisticregression import logisticregression


class multiclass:
    def __init__(self,k,regularization="No",lmbda=0.1):
        self.k = k
        self.regressors = []
        for i in range(k):
            lr1 = logisticregression(regularization=regularization,lmbda=lmbda)
            self.regressors.append(lr1)

    def fit(self,X,y,intercept_addition=True,n_iterations = 100,learning_rate = 0.01,use_autograd=False):
        a1 = list(set(list(y)))
        assert(len(a1)==self.k)
        a1.sort()
        vals = dict()
        vals_rev = dict()
        for i in range(len(a1)):
            vals[a1[i]] = i
            vals_rev[i] = a1[i]
        self.vals = vals
        self.vals_rev = vals_rev
        
        for i in range(self.k):
            yin = np.copy(y)
            yin[y==a1[i]] = 1
            yin[y!=a1[i]] = 0
            
            self.regressors[i].fit(X,yin,intercept_addition=intercept_addition,n_iterations=n_iterations,learning_rate=learning_rate,use_autograd=use_autograd)
    
    def predict_helper(self,i,X,intercept_addition=True):
        if(intercept_addition==True):
            column = np.ones((X.shape[0],1))
            X = np.hstack((column,X))
        # print("prediction started")
        # print(self.weights.shape)
        # print(X.shape)
        a = np.dot(X,self.regressors[i].weights)
        a = 1.0/(1+np.exp(-a))
        return a

    def predict(self,X,intercept_addition=True):
        regressor_results = []
        predictions =[]
        for i in range(self.k):
            regressor_results.append(self.predict_helper(i,X,intercept_addition=intercept_addition))

        for i in range(regressor_results[0].shape[0]):
            max_i = -1
            max_prob = -1
            for j in range(len(regressor_results)):
                if(regressor_results[j][i]>max_prob):
                    max_prob = regressor_results[j][i]
                    max_i = j
            predictions.append(self.vals_rev[max_i])
                        
        return np.array(predictions)


###### checking multi class####
# X,y = load_breast_cancer(return_X_y=True)
# sc = StandardScaler().fit(X)
# X = sc.transform(X)
digits = load_digits()
data = digits.images
n = len(data)

X = data.reshape((n,-1))

y = digits.target

sc = StandardScaler().fit(X)
X = sc.transform(X)

############ first and second part ###############
# mk = multiclass(len(list(set(list(y)))))
# mk.fit(X,y)
# mk.fit(X,y)

# prediction = mk.predict(X)

# print(((prediction==y).sum()/y.shape[0])*100)
###########################################



############# part 3 cross validation and confusion matrix ###############

# kf = KFold(n_splits=4)
# acc =0
# conf_avg = np.zeros([10,10])
# for train, test in kf.split(X):
#     X_train,y_train,X_test,y_test = X[train],y[train],X[test],y[test]
#     mk = multiclass(len(list(set(list(y)))))
#     mk.fit(X_train,y_train,intercept_addition=True,n_iterations=100,learning_rate=0.01)
#     prediction = mk.predict(X_test)
#     acc += ((prediction==y_test).sum()/(y_test.shape[0]))*100
#     conf_matrix = confusion_matrix(y_test,prediction)
#     conf_avg+=conf_matrix
#     # print(conf_matrix)    
    
# conf_avg = conf_avg/4
# print(conf_avg)
# print("Average accuracy:" ,acc/4)


# [[43.5   0.    0.    0.    0.    0.25  0.25  0.25  0.25  0.  ]
#  [ 0.   41.25  0.5   0.5   0.25  0.25  0.25  0.    1.25  1.25]
#  [ 0.5   0.25 43.    0.25  0.    0.    0.    0.    0.25  0.  ]
#  [ 0.25  0.25  1.5  39.75  0.    0.75  0.    0.75  1.75  0.75]
#  [ 0.    0.75  0.25  0.   42.    0.    0.25  0.5   0.5   1.  ]
#  [ 0.25  0.5   0.25  0.    0.25 43.25  0.25  0.25  0.    0.5 ]
#  [ 0.75  0.5   0.    0.    0.5   0.   43.5   0.    0.    0.  ]
#  [ 0.    0.    0.    0.25  1.    0.    0.   42.75  0.5   0.25]
#  [ 0.25  3.25  1.25  0.25  0.25  0.75  1.    0.   35.75  0.75]
#  [ 0.75  2.25  0.25  1.    0.75  1.    0.    1.25  2.   35.75]]
# Average accuracy: 91.3739173471913


##################################################




################ part 4 PCA black box ###################
print(X.shape)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
print(X.shape)
print(y.shape)
############ plotting ####################
x1 = X[:,0:1]
y1 = X[:,1:2]
labels = y
colors = ['red','green']
plt.scatter(x1, y1, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()


#########################################################


################    Q4       ######################

#### space complexity = O(number of parameters*number of samples)
###### training time complexity = O(number of samples * number_of_parameters)
#######################################