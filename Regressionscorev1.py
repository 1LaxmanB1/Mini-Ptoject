import numpy as np
import matplotlib.pyplot as plt
import math
def regressionscorev1(model,X_Test,Y_true,regname,dataname):

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

    #______RMSE CALCULATION________
    mse = np.square(np.subtract(Y_true,y_predict)).mean()
    rmse = math.sqrt(mse)
    # _____R^2 CALCULATION USING SCI-KIT LIBRARY FUNCTION_______
    R2L = model.score(X_Test,Y_true)

    # _____REGRESSION PLOT_______________________
    xplot = y_predict.copy()
    yplot = Y_true.copy()
    p1 = max(max(y_predict), max(Y_true))
    p2 = min(min(y_predict), min(Y_true))

    fig = plt.figure(figsize=(8,6))
    plt.scatter(x=xplot, y=yplot, c='crimson')
    plt.plot([p1,p2],[p1,p2],'b-')
    plt.xlabel("Predicted value")
    plt.ylabel("Actual value")
    plt.title("Regression plot on " + dataname +" using " + regname +" , its R^2 is " + str(R2))
    plt.legend(["(Predicted,Actual value) as points",'Regression Line'],loc ="lower right")
    #plt.ion()
    # plt.show()  # To show the plot

    Scoremetrics = {
                    "R^2" : R2,
                    "Accuracy" : Accuracy,
                    "Average absolute Error" : avgerr,
                    "Root Mean Squared Error " : rmse
                    }

    return Scoremetrics,fig