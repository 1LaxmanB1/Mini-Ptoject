from datagen import X_train,X_test,Y_train,Y_test
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt


# _______RANDOM FOREST REGRESSION__________________

rfreg = RandomForestRegressor(min_samples_split=10,min_samples_leaf=10)
rfreg.fit(X_train,Y_train)
ytest_predict = rfreg.predict(X_test)


