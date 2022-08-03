from datagen import X_train,X_test,Y_train,Y_test
from sklearn import svm
from Regressionscorev1 import regressionscorev1
import numpy as np
import matplotlib.pyplot as plt


# _______SUPPORT VECTOR REGRESSION__________________

ker='rbf'
eps=0.1
reg=0.7
svrreg = svm.SVR(kernel=ker,epsilon=eps,C=1)
svrreg.fit(X_train,Y_train)
ytest_predict = svrreg.predict(X_test)

# _______REGRESSION SCORE USING SVR__________________

regressionscorev1(Y_test,ytest_predict,'SVR')