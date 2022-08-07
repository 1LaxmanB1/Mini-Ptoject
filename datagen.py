from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from skfeature.function.similarity_based import fisher_score
import seaborn as sns
from mlxtend.feature_selection import ExhaustiveFeatureSelector
# from sklearn.linear_model import LinearRegression as lr
from Featureselectionv1 import featureselection

dtyp = 0  #Decision variable to choose between Real world / Toy dataset


if dtyp == 1:

    # __________ IMPORTING HOUSING DATASETS FROM SCI-KIT LEARN (REAL-WORLD DATASET)___________

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
    #print(type(inputdata))  # The input features are a n-dimensional numpy array
    print("Total dataset size : ")
    print(inputdata.shape)  # The dimension of the array is checked with in par to the documentation
    outputtarget = housesdata['target'].copy()  # Contains the house value / price as the output data
    feature_list = housesdata['feature_names'].copy()

    #_______PANDAS DATAFRAME CREATION__________

    hsdfv1 = pd.DataFrame(inputdata)
    hsdfv1.columns = feature_list
    hsdfv1['HouseValue'] = outputtarget
    print(hsdfv1)
    inputdatav2 = hsdfv1[['MedInc','AveRooms']]
    dataf=hsdfv1


    # _____DATA VISUALIZATION (Housing)________

    hsdfv1.plot(kind="scatter",
            x="Longitude",
            y="Latitude",
            alpha=0.5,
            s=hsdfv1["Population"]/100,
            label="Population",
            c="HouseValue",
            cmap=plt.get_cmap("jet"),
            colorbar=True)
    plt.legend()
    plt.show()


else :

    # __________IMPORTING DIABETES DATASET FROM SCI-KIT LEARN (TOY DATASET)____________

    diadata = datasets.load_diabetes()
    inputdata =diadata['data'].copy()
    print("Total dataset size : ")
    print(inputdata.shape)  # The dimension of the array is checked with in par to the documentation
    outputtarget = diadata['target'].copy() # Contains the house value / price as the output data
    feature_list = diadata['feature_names'].copy()

    #_______PANDAS DATAFRAME CREATION__________

    ddfv1 = pd.DataFrame(inputdata)
    ddfv1.columns = feature_list
    ddfv1['Target'] = outputtarget
    print(ddfv1)
    inputdatav2 = ddfv1[['bmi','bp','s5','s6']]    #This was manually created after looking the results of feature selction fn

    dataf=ddfv1





# ________SPLITTING TRAINING AND TESTING DATA______________

ts = 0.3 # Choosing percentage of splitting for testing data ranges from (0 1)
slice = 500 # No. of data points to be taken with no staring point specified
X_train,X_test,Y_train,Y_test = train_test_split(dataf.iloc[1:slice,:-1],outputtarget[1:slice],test_size = ts)
# print(inputdata[1:slice,1].reshape(-1,1))
print("Training set size : ")
print(X_train.shape)
print("Testing set size : ")
print(X_test.shape)

# _______FEATURE SELECTION___________
print(dataf)
featureselection(X_train, Y_train, feature_list, dataf)

