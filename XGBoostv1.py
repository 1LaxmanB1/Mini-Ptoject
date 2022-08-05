from datagen import X_train,X_test,Y_train,Y_test
from xgboost import XGBRegressor
from Regressionscorev1 import regressionscorev1
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from pprint import pprint

# ______CREATING SEARCH SPACE FOR SOME IMPORTANT HYPER-PARAMETERS IN BOOST______


n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]  # Number of trees in random forest
learning_rate = [0.1, 0.01, 0.3, 0.5]
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]  # Maximum no. of levels in a tree

random_grid = \
    {
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    }

# _______ BASIC XGBOOST MODEL__________________

xgbreg = XGBRegressor()
print('Parameters used by base model (default values) : \n')
pprint(xgbreg.get_params())

print("\n Search space for Random-Search : \n")
pprint(random_grid)  # To know about the search space we are going
# to randomly look for the best choice of hyper-parameters


# _________RANDOM SEARCH ON XGBOOST______________

'''Search across 100 random combinations of given space of hyper-parameters
   ,use all available cores, & perform 5 fold cross-validation '''

xgb_random = RandomizedSearchCV(estimator=xgbreg, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)

xgb_random.fit(X_train, Y_train)  # Fit the random search model
print("The best set of hyper-parameters from random-search : \n")
pprint(xgb_random.best_params_)


xgbbase = XGBRegressor(random_state=42)
xgbbase.fit(X_train, Y_train)
xgbbase_accuracy=regressionscorev1(xgbbase, X_test, Y_test, 'XGBoost Base model')
print("Score metrics for XGBoost Base model : \n")
pprint(xgbbase_accuracy)


xgbbest = xgb_random.best_estimator_
xgbbest_accuracy = regressionscorev1(xgbbase, X_test, Y_test, 'XGBoost Best model')
print("Score metrics for XGBoost Best model via Random Search: \n")
pprint(xgbbest_accuracy)
