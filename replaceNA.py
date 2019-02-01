
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



def dropNA(df, axis = 'rows', how = 'all'):
    try:
        rows = df.shape[0]
        goals = []
        
        if axis == 'rows':
            columns = df.shape[1]
            for i in df.index:
                if how == 'all':
                    if sum(df.loc[i].isna()) == columns:
                        goals.append(i)
                elif how == 'any': 
                    if sum(df.loc[i].isna()) != 0:
                        goals.append(i)
            return df.drop(goals)

        elif axis == 'columns':
            columns = df.columns
            for column in columns:
                if how == 'all':
                    if sum(df[column].isna()) == rows:
                        goals.append(column)
                elif how == 'any':
                    if sum(df[column].isna()) != 0:
                        goals.append(column)
            return df.drop(columns = goals)

        else:
            return df
        
    except Exception as e:
        print(e)




def replace(df, columns, value = 'mean'):
    try:
        for column in columns:   
            if value == 'mean':   
                goal = df[column].mean()
            elif value == 'median':
                goal = df[column].median()
            elif value == 'mode':
                goal = df[column].mode()[0]

            for i in df.index:
                if pd.isnull(df.loc[i, column]):
                    df.loc[i, column] = goal
        return df
    except Exception as e:
        print(e)
        


def replaceRegression(df, X, y):
    try:
        indexes = [i for i in df.index if not pd.isnull(df.loc[i, X])]
        X_train = np.array((df.loc[indexes, X])).reshape(-1,1)
        y_train = np.array((df.loc[indexes, y])).reshape(-1,1)
        reg = LinearRegression().fit(X_train, y_train)
        for i in df.index:
            if pd.isnull(df.loc[i, X]):
                df.loc[i, X] = reg.predict(df.loc[i, y])[0][0]
        return df
    except Exception as e:
        print(e)


def standartization(df, columns):
    try:
        for column in columns:
            mean = df[column].mean()
            std = df[column].std()
            for i in df.index:
                df.loc[i, column] = df.loc[i, column] * mean / std
        return df
    except Exception as e:
        print(e)


def scaling(df, columns):
    try:
        for column in columns:
            Max = max(df[column])
            Min = min(df[column])
            for i in df.index:
                df.loc[i, column] = (df.loc[i, column] - Min) / (Max - Min)
        return df
    except Exception as e:
        print(e)



def KNN (df, trainData, testData, k, numberOfClasses):
    try:
        def dist (a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        testLabels = []
        for testPoint in testData:
            testDist = [ [dist(testPoint, trainData[i][0]), trainData[i][1]] for i in range(len(trainData))]
            stat = [0 for i in range(numberOfClasses)]
            for d in sorted(testDist)[0:k]:
                stat[d[1]] += 1
            testLabels.append( sorted(zip(stat, range(numberOfClasses)), reverse=True)[0][1] )
        return testLabels
    except Exception as e:
        print(e)

def fillKNN(df, target, trainData, testData, numberOfClasses, k):
    try:
        for i in df.index:
                if pd.isnull(df.loc[i, target]):
                    df.loc[i, target] = KNN(df, trainData, testData, k, numberOfClasses)
        return df
    except Exception as e:
        print(e)

