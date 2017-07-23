# Import the Pandas library
import pandas as pd
import numpy as np

# Load the train and test datasets to create two DataFrames
train_url = "train.csv"
train = pd.read_csv(train_url)
test_url = "test.csv"
test = pd.read_csv(test_url)
gender_url = "gender_submission.csv"
gender = pd.read_csv(gender_url)

pd.options.mode.chained_assignment = None 

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

age_median = train["Age"].median()
age_median_test = test["Age"].median()

train["Age"] = train["Age"].fillna(age_median)
test["Age"] = test["Age"].fillna(age_median_test)


# Create a new array with the added features: features_two
features = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
target = train["Survived"].values

features_survived = train[train["Survived"]==1]
features_dead = train[train["Survived"]==0]

fs = features_survived[["Age","Sex"]].values
fd = features_dead[["Age","Sex"]].values

target_test = gender["Survived"]

test.Fare[152] = test.Fare.median()

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

import matplotlib.pyplot as plt
plt.plot(fs, fd, c ='r')
#plt.scatter(fd, fd)
plt.show()

from sklearn.metrics import accuracy_score
from time import time
from sklearn.svm import SVC
clf = SVC(C=10.0,kernel="rbf")
t0 = time()
clf.fit(features, target)
print "training time:", round(time()-t0, 3), "s"

print "SVM.score = ", clf.score(features,target)

t0 = time()
pred = clf.predict(test_features)
print "predicting time:", round(time()-t0, 3), "s"
print "SVM Accuracy: ",accuracy_score(target_test,pred)


