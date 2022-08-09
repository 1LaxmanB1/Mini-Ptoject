from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import ExhaustiveFeatureSelector
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from skfeature.function.similarity_based import fisher_score
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector
# from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

def featureselection(inputdata,outputtarget,feature_list,hsdfv1):




    # _______Feature selection via Information gain___________

    importances = mutual_info_regression(inputdata,outputtarget)
    feat_importances = pd.Series(importances, feature_list)
    plt.subplot(1,2,1)
    feat_importances.plot(kind = 'barh', color = 'teal')
    print(feature_list)
    plt.title("Feature importance graph via Information gain")
    plt.ylabel("Features")
    plt.xlabel("Score")
    # plt.show()

    # # _______Feature selection via Fisher score-------------------
    # print(inputdata)
    # print(outputtarget)
    # ranks = fisher_score.fisher_score(inputdata,outputtarget)
    #
    # print(ranks)
    # print(feature_list)
    # feat_importances_fs = pd.Series(ranks, feature_list)
    # plt.subplot(2,2,2)
    # feat_importances_fs.plot(kind='barh', color = 'teal')
    # print(feature_list)
    # plt.title("Feature importance graph via Fisher Score")
    # plt.ylabel("Features")
    # plt.xlabel("Score")


    # ________Feature selection via correlation coefficent____________

    corr = hsdfv1.corr() #Correlation matrix
    plt.subplot(1,2,2)
    sns.heatmap(corr,annot = True)
    plt.title ("Correlation Heat-map to identify relationship between features")
    plt.show()


    # # __________Sequential Exhaustive Feature Selection______________
    #
    # efs = ExhaustiveFeatureSelector(RandomForestRegressor(),  min_features=2,max_features=8,scoring='roc_auc',cv=2)
    # efs=efs.fit(inputdata,outputtarget)
    # efsfeatures = feature_list[list(efs.best_idx_)]
    # print(efsfeatures)
    # print(efs.best_score_)






