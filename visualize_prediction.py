# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 20:38:04 2018

@author: SARTHAK BABBAR
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:12:13 2018

@author: SARTHAK BABBAR
"""


import pandas as pd 
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
 
style.use('ggplot')

df= pd.read_csv("dataset_1.csv")
dp=pd.read_csv("predict.csv")
#df=df[['sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']]

print(df.shape)

df=df.dropna(axis=0,how='any')
print(df.shape)
X=np.array(df.drop(['G3'],1))
X_predict=np.array(dp.drop(['G3'],1))
X_predict=preprocessing.scale(X_predict)
X=preprocessing.scale(X)

y=np.array(df['G3'])
g2=np.array(df['G2'])
g2_predict=np.array(dp['G2'])
clf=LinearRegression()
clf.fit(X,y)

y_pred = clf.predict(X_predict)

np.savetxt('predictions.csv',y_pred, delimiter=',')

    
plt.scatter(g2, y,  color='black')
plt.scatter(g2_predict,y_pred,color='blue',lw=3)
plt.legend(loc=4)
plt.xlabel('2nd Term Marks')
plt.ylabel('Final Marks')
plt.show()
#we are not getting a straight line because there are multiple y for the same x since the other
#features other than G2 can be different




