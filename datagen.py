from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# __________ IMPORTING DATASETS FROM SCI-KIT LEARN___________

"""
There are 20640 samples in total with 8 features but 
we'll be using only first 500 samples for now

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


