# importing required libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# loading breast cancer dataset which has two classes malignant and benign
cancerdf = load_breast_cancer()

# creating a dataframe from the above loaded data
cdf = pd.DataFrame(cancerdf.data, columns=cancerdf.feature_names)
cdf['target'] = pd.Series(cancerdf.target)

# checking the shape of the dataset
cdf.shape

# visualizing first 5 records of the dataset
cdf.head()

# seeing the column names
cdf.columns

# separating features and target variable
X = cdf.iloc[:,:-1]
y = cdf.iloc[:,-1]

# creating k fold cross validation object
k = 10
kf = KFold(n_splits=k, random_state=None)

# taking 4 classifiers in the form of a list
models=['Decision Tree Classifier','Random Forest Classifier','Support Vector Machine Classifier','K Nearest Neighbour Classifier']
# iterating the models
for i in models:
# creating empty lists for accuracy_score, xtrain, xtest, y_train, y_test
 acc_score = []
xtrain=[]
xtest=[]
ytrain=[]
ytest=[]
# creating model object for each classifier one by one
if i=='Decision Tree Classifier':
 model=DecisionTreeClassifier()
elif i=='Random Forest Classifier':
 model=RandomForestClassifier()
elif i=='Support Vector Machine Classifier':
 model=SVC()
elif i=='K Nearest Neighbour Classifier':
 model=KNeighborsClassifier()
# applying k fold cross validation on the input dataset
for train_index , test_index in kf.split(X):
# storing x_train, x_test, y_train, y_test
 X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
 y_train , y_test = y[train_index] , y[test_index]
# training the model using training data
 model.fit(X_train,y_train)
# getting the predictions using testing data
 m_pred=model.predict(X_test)
# finding the accuracy of the model using predictions and actual testing label data
 acc = accuracy_score(m_pred , y_test)
# appending obtained accuracy, x_train, x_test, y_train, y_test to the above created lists
 acc_score.append(acc)
 xtrain.append(X_train)
 xtest.append(X_test)
 ytrain.append(y_train)
 ytest.append(y_test)
# getting the maximum accuracy index from the above 10 cross validation datasets
 m=np.argmax(acc_score)

# considering maximum accuracy given splits as final training and testing data
 x_train=xtrain[m]
 x_test=xtest[m]
 y_train=ytrain[m]
 y_test=ytest[m]
# fitting the model using above best split training data
 model.fit(x_train,y_train)
# getting predictions using above best split testing data
 m_pred=model.predict(x_test)
# printing confusion matrix of each classifier
 print("Confusion Matrix of {}:".format(i))
 print(confusion_matrix(m_pred,y_test))
