import pandas as pd

#SVM 룰루!
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

print('1. Training Data Set Score ')
print(svclassifier_poly.score(X_train, y_train))
print('2. Test Data Set Score ')
print(svclassifier_poly.score(X_test, y_test))
print('3. Classifier Parameter ')
print(svclassifier_poly.get_params())


#Evaluating
from sklearn.metrics import classification_report, confusion_matrix
print('4. Confusion Matix')
print(confusion_matrix(y_test, y_pred))
print('5. Classification Report ')
print(classification_report(y_test, y_pred))


test_pred = svclassifier_poly.predict([[ -3.7503,-13.4586,17.5932,-2.7771]])
print("test(-3.7503,-13.4586,17.5932,-2.7771, class =1)", test_pred)

test_pred = svclassifier_poly.predict([[0.40614,1.3492,-1.4501,-0.65949]])
print("test(0.40614,1.3492,-1.4501,-0.65949, (1) )", test_pred)

test_pred = svclassifier_poly.predict([[4.1665,-0.5449,0.23448,0.27843]])
print("test(4.1665,-0.5449,0.23448,0.27843, (0) )", test_pred)
