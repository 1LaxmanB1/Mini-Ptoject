from datagen import X_train,X_test,Y_train,Y_test
from sklearn import svm
from Regressionscorev1 import regressionscorev1
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

# ______CREATING SEARCH SPACE FOR SOME IMPORTANT HYPER-PARAMETERS IN RANDOM FOREST______

kernel = ['linear', 'poly', 'rbf', 'sigmoid']
C=[0.25, 0.5, 0.75, 1 ]
epsilon = [0.00001, 0.1, 0.25, 0.5, 1]
degree = [3, 5, 7, 9]
random_grid_svr = \
    {
    'kernel': kernel,
    'C': C,
    'epsilon': epsilon,
    'degree' : degree
    }

# _______ BASIC RANDOM FOREST MODEL__________________

svreg = svm.SVR()
print('Parameters used by base model (default values) : \n')
pprint(svreg.get_params())

print("\n Search space for Random-Search : \n")
pprint(random_grid_svr)  # To know about the search space we are going
# to randomly look for the best choice of hyper-parameters

# _________RANDOM SEARCH ON SVR______________

'''Search across 100 random combinations of given space of hyper-parameters
   ,use all available cores, & perform 5 fold cross-validation '''

svr_random = RandomizedSearchCV(estimator=svreg, param_distributions=random_grid_svr, n_iter=100, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)

svr_random.fit(X_train, Y_train)  # Fit the random search model
print("The best set of hyper-parameters from random-search : \n")
pprint(svr_random.best_params_)


svrbase = svm.SVR()
svrbase.fit(X_train,Y_train)
svrbase_accuracy=regressionscorev1(svrbase,X_test,Y_test,'SVR Base model')
print("Score metrics for SVR Base model : \n")
pprint(svrbase_accuracy)


svrbest = svr_random.best_estimator_
svrbest_accuracy=regressionscorev1(svrbest,X_test,Y_test, 'RF Best model')
print("Score metrics for SVR Best model via Random Search: \n")
pprint(svrbest_accuracy)




