from select import select
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split



def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]



AllTrees = []

def Build_RandomForest(X, y ,numof_tree = 20 ):
     
      for i in range(numof_tree):
        model = tree.DecisionTreeClassifier()
        X_samp, y_samp = bootstrap_sample(X, y)
        model.fit(X_samp, y_samp)
        AllTrees.append(model)



def Pre_RandomForest(X_test):
    
        Main_tree = []
        for i in range(len(X_test)):
          ar = []
          for k in range(len(AllTrees)):
              
            pr_val = AllTrees[k].predict([X_test[i]])
            
            ar.append(pr_val[0])
            
          Main_tree.append(ar)
        
        return(Main_tree)    



data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


Build_RandomForest(X_train,y_train)
y_pred = Pre_RandomForest(X_test)

Pre_Label = []


for i in range(len(y_pred)):
  sin_val =  max(set(y_pred[i]), key = y_pred[i].count)
  Pre_Label.append(sin_val)

def accuracy(y_true, y_pred):

    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


acc = accuracy(y_test, Pre_Label)

print("Accuracy:", acc)
