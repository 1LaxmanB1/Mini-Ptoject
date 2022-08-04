import numpy as np
import matplotlib.pyplot as plt

def regressionscorev1(model,X_Test,Y_true,regname):

    # ________Model prediction__________

    y_predict=model.predict(X_Test)

    # ______R^2 CALCULATION FOR TESTING DATA_____________________
    u = ((Y_true - y_predict) ** 2).sum()
    v = ((Y_true - np.mean(Y_true)) ** 2).sum()
    R2 = (1 - (u / v))
    errors = abs(Y_true-y_predict)
    mape = 100 * np.mean(errors/Y_true)
    avgerr = np.mean(errors)
    Accuracy = 100 - mape

    # _____REGRESSION PLOT_______________________
    xplot = y_predict.copy()
    yplot = Y_true.copy()

    plt.scatter(x=xplot, y=yplot)
    plt.xlabel("Predicted House value in $ 100,000 ")
    plt.ylabel("Actual House value in $ 100,000 ")
    plt.title("Regression plot using " + regname +" & its R^2 is " + str(R2))
    #plt.ion()
    plt.show()  # To show the plot

    Scoremetrics = {
                    "R^2" : R2,
                    "Accuracy" : Accuracy,
                    "Average abs Error" : avgerr
                    }

    return Scoremetrics