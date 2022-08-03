from datagen import X_train, X_test, Y_train, Y_test
from RandomForestv1 import ytest_predict
import numpy as np
import matplotlib.pyplot as plt

# ______R^2 CALCULATION FOR TESTING DATA_____________________
u = ((Y_test - ytest_predict) ** 2).sum()
v = ((Y_test - np.mean(Y_test)) ** 2).sum()
R2 = (1 - (u / v))

# _____REGRESSION PLOT_______________________
xplot = ytest_predict.copy()
yplot = Y_test.copy()

plt.scatter(x=xplot, y=yplot)
plt.xlabel("Predicted House value in $ 100,000 ")
plt.ylabel("Actual House value in $ 100,000 ")
plt.title("Regression plot using Random Forest & its R^2 is " + str(R2))
plt.show()  # To show the plot
