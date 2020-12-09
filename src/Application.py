#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 01:29:24 2020

@author: lasitha
"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


data = pd.read_csv('../data/train.csv')

print(data.shape)



plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

#check the distribution of SalePrice
print(data.SalePrice.describe())

#plot the histrogram of the SalePrice
plt.hist(data.SalePrice, color='blue')
plt.show()

numeric_features = data.select_dtypes(include=[np.number])
print(numeric_features.dtypes)



