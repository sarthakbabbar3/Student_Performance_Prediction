from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing



url = "dataset_1.csv"
df = pd.read_csv(url)

X=np.array(df.drop(['G3'],1))
y=np.array(df['G3'])
print(X.shape)
X=preprocessing.scale(X)
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)

print(clf.feature_importances_)  

np.savetxt('treebased.csv',clf.feature_importances_, delimiter=',')

