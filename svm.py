import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#data processing
bankdata = pd.read_csv("D:\Program\Pycharm\Project/bill_authentication.csv")
bankdata.shape
bankdata.head()

X= bankdata.drop('Class', axis=1)
y= bankdata['Class']

#split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

#Training the alg
from sklearn.svm import SVC

# linear kernel
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#poly
svclassifier_poly = SVC(kernel='poly', degree=8)
svclassifier_poly.fit(X_train, y_train)

#Gaussian
svclassifier_rbf = SVC(kernel='rbf')
svclassifier_rbf.fit(X_train, y_train)


#predictions
y_pred = svclassifier_poly.predict(X_test)

#Evaluating
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
