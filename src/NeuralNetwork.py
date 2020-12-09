#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 02:35:13 2020

@author: lasitha
"""


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor


train_data = pd.read_csv('../data/train.csv')

test_data = pd.read_csv('../data/test.csv')

train_missing = train_data.isna().sum() / train_data.shape[0]

train_missing = train_missing[train_missing > 0.7]

print(train_missing.head())


test_missing = test_data.isna().sum() / test_data.shape[0]
test_missing = test_missing[test_missing > 0.7]

print(test_missing.head())

#Drop columsn with null values precentage > 70
train_data.drop(train_missing[train_missing > 0.7].index, axis = 1, inplace = True)
test_data.drop(test_missing[test_missing > 0.7].index, axis = 1, inplace = True)


#manage missing values
train_data['LotFrontage'].interpolate(axis=0, inplace=True)
train_data[['MasVnrType']].fillna('None', inplace=True)
train_data.dropna(subset=['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'], inplace=True)
train_data.dropna(subset=['BsmtQual', 'BsmtCond', 'BsmtFinType1'], inplace=True)
train_data.dropna(subset=['MasVnrType', 'MasVnrArea'], inplace=True)
train_data.drop('FireplaceQu', axis=1, inplace=True)
train_data['BsmtExposure'].fillna('No', inplace=True)
train_data['BsmtFinType2'].fillna('Unf', inplace=True)
train_data['Electrical'].fillna('SBrkr', inplace=True)

#manage the missing values in the test dataset 

test_data.fillna(method='ffill', inplace=True)
test_data.fillna(method='bfill', inplace=True)


# Copy Train data excluding target
trainData_Copy = train_data.drop(['SalePrice', 'Id'], axis=1).copy()
testData_Copy = test_data.drop('Id', axis=1).copy()
# Combine Train and test for One Hot Encoding
combined_Data = pd.concat([trainData_Copy,testData_Copy], keys = [0,1])
# Do One Hot Encoding for categorical features
combined_Data = pd.get_dummies(combined_Data)
# Separate Train data and test data
X_train = combined_Data.xs(0)
X_test = combined_Data.xs(1)
y_train = train_data["SalePrice"]


mlp = MLPRegressor()
mlp.fit(X_train, y_train)
mlp_predict = mlp.predict(X_test)








