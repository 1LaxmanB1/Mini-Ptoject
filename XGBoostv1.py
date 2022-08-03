from datagen import X_train,X_test,Y_train,Y_test
import xgboost as xgb
from Regressionscorev1 import regressionscorev1
import numpy as np
import matplotlib.pyplot as plt

# _____BRINGING THE DATASET INTO XGBOOST D MATRIX FORMAT______





# _______SUPPORT VECTOR REGRESSION__________________



# _______REGRESSION SCORE USING SVR__________________

regressionscorev1(Y_test,ytest_predict,'SVR')