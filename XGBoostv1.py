from datagen import X_train,X_test,Y_train,Y_test
from xgboost import XGBRegressor
from Regressionscorev1 import regressionscorev1
import numpy as np
import matplotlib.pyplot as plt

# _______SUPPORT VECTOR REGRESSION__________________



boost = XGBRegressor()
boost.fit(X_train,Y_train)
ytest_predict = boost.predict(X_test)

# _______REGRESSION SCORE USING SVR__________________

regressionscorev1(Y_test,ytest_predict,'XGBoost')