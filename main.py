from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# __________ IMPORTING DATASETS FROM SCI-KIT LEARN___________

"""
inputdata consists of :
MedInc     - median income in block group
HouseAge   - median house age in block group
AveRooms   - average number of rooms per household
AveBedrms  - average number of bedrooms per household
Population - block group population
AveOccup   - average number of household members
Latitude   - block group latitude
Longitude  - block group longitude
"""

housesdata = datasets.fetch_california_housing()
inputdata = housesdata['data'].copy()
print(type(inputdata))  # The input features are a n-dimensional numpy array
print(inputdata.shape)  # The dimension of the array is checked with in par to the documentation
outputtarget = housesdata['target'].copy()  # Contains the house value / price as the output data


# ________SPLITTING TRAINING AND TESTING DATA______________

ts = 0.4 # Choosing percentage of splitting for testing data ranges from (0 1)
slice = 500 # No. of data points to be taken with no staring point specified
X_train,X_test,Y_train,Y_test = train_test_split(inputdata[1:slice],outputtarget[1:slice],test_size = ts)
#print(X_train.shape)
# _______RANDOM FOREST REGRESSION__________________

rfreg = RandomForestRegressor(min_samples_split=10,min_samples_leaf=10)
rfreg.fit(X_train,Y_train)
ytest_predict = rfreg.predict(X_test)
#print(ytest_predict.shape)
#print(Y_test.shape)

# ______R^2 CALCULATION_____________________
u=((Y_test-ytest_predict)**2).sum()
v=((Y_test-np.mean(Y_test))**2).sum()
R2 = (1-(u/v))


# _____REGRESSION PLOT_______________________
xplot = ytest_predict.copy()
yplot = Y_test.copy()

plt.scatter(x=xplot,y=yplot)
plt.xlabel("Predicted House value in $ 100,000 ")
plt.ylabel("Actual House value in $ 100,000 ")
plt.title("Regression plot using Random Forest & its R^2 is " + str(R2))
plt.show() # To show the plot