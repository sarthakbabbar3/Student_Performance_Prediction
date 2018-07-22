# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 19:57:55 2018

@author: SARTHAK BABBAR
"""

import pandas as pd 
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

df= pd.read_csv("dataset_1.csv")
df=df.dropna(axis=0,how='any')

print(df.shape)
X=np.array(df.drop(['G3'],1))


y=np.array(df['G3'])

sm = SMOTE()

X_resampled,y_resampled = sm.fit_sample(X, y)

print(X_resampled.shape)

np.savetxt('smotex.csv',X_resampled, delimiter=',')
np.savetxt('smotey.csv',y_resampled, delimiter=',')
   


