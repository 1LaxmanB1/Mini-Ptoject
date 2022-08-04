from datagen import X_train, X_test, Y_train, Y_test
from sklearn.ensemble import RandomForestRegressor
from Regressionscorev1 import regressionscorev1
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

# ______CREATING SEARCH SPACE FOR SOME IMPORTANT HYPER-PARAMETERS IN RANDOM FOREST______


n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]  # Number of trees in random forest
max_features = [1.0, 'sqrt', 'log2']  # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]  # Maximum no. of levels in a tree
min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
min_samples_leaf = [1, 3, 5]  # Minimum number of samples required at each leaf node
boostrap = [True, False]  # Method of selecting samples for training each tree

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': boostrap
}

# _______ BASIC RANDOM FOREST MODEL__________________

rfreg = RandomForestRegressor()
print('Parameters used by base model (default values) : \n')
pprint(rfreg.get_params())

print("\n Search space for Random-Search : \n")
pprint(random_grid)  # To know about the search space we are going
# to randomly look for the best choice of hyperparameters

# _________RANDOM SEARCH ON RANDOM FOREST______________

'''Search across 100 random cobinations of given space of hyper-parameters
   ,use all available cores, & perform 5 fold cross-validation '''

rf_random = RandomizedSearchCV(estimator=rfreg, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)

rf_random.fit(X_train, Y_train)  # Fit the random search model
print("The best set of hyper-parameters from random-search : \n")
pprint(rf_random.best_params_)


rfbase = RandomForestRegressor(random_state=42)
rfbase.fit(X_train,Y_train)
rfbase_accuracy=regressionscorev1(rfbase,X_test,Y_test,'RF Base model')
print("Score metrics for Random Forest Base model : \n")
pprint(rfbase_accuracy)


rfbest = rf_random.best_estimator_
rfbest_accuracy=regressionscorev1(rfbest,X_test,Y_test, 'RF Best model')
print("Score metrics for Random Forest Best model via Random Search: \n")
pprint(rfbest_accuracy)
