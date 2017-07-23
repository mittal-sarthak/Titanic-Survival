# Import the Pandas library
import pandas as pd


# Load the train and test datasets to create two DataFrames
train_url = "train.csv"
train = pd.read_csv(train_url)
test_url = "test.csv"
test = pd.read_csv(test_url)
test_pred_url = "gender_submission.csv"
test_pred = pd.read_csv(test_pred_url)
#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())
pd.options.mode.chained_assignment = None  # default='warn'

# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts())

# As proportion
print(train["Survived"].value_counts(normalize = True))

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))

# Normalized female survival
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))

# Create the column Child and assign to 'NaN'
train["Child"] = 0

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
print(train["Child"])


# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))


# Create a copy of test: test_one
test_one = test

# Initialize a Survived column to 0
test_one["Survived"] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one["Survived"][test_one["Sex"] == "female"] = 1
print(test_one["Survived"])



# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree


# Convert the male and female groups to integer form
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

#Print the Sex and Embarked columns
print (train["Sex"])
print (train["Embarked"])


# Print the train data to see the available features
print(train)
age_median = train["Age"].median()
age_median_test = test["Age"].median()

train["Age"] = train["Age"].fillna(age_median)
test["Age"] = test["Age"].fillna(age_median_test)

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one,target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print("Decision Tree",my_tree_one.score(features_one,target))


# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
print (test_features)
# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)
print (my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])




# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
test_features_two = test[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two,target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))

two_pred = my_tree_two.predict(test_features_two)
# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution3 = pd.DataFrame(two_pred, PassengerId, columns = ["Survived"])

# Write your solution to a csv file with the name my_solution.csv
my_solution3.to_csv("my_solution_3.csv", index_label = ["PassengerId"])



# Create train_two with the newly defined feature
train_two = train.copy()
test_two = test.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1
test_two["family_size"] = test_two["SibSp"] + test_two["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
test_features_three = test_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three,target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))





# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=10, n_estimators = 1000 , random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
target_test = test_pred["Survived"]
print(len(pred_forest))
print(len(target_test))
from sklearn.metrics import accuracy_score
my_score = accuracy_score(target_test,pred_forest)
print ("forest score", my_score)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution2 = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

# Write your solution to a csv file with the name my_solution.csv
my_solution2.to_csv("my_solution_2.csv", index_label = ["PassengerId"])


#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_forest, target))


from time import time
from sklearn.svm import SVC
clf = SVC(C=1,kernel="linear")
t0 = time()
clf.fit(features_two, target)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred_svm = clf.predict(test_features)
print "predicting time:", round(time()-t0, 3), "s"
print pred_svm
print ("svm"),accuracy_score(target_test,pred_svm)

PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution4 = pd.DataFrame(pred_svm, PassengerId, columns = ["Survived"])

# Write your solution to a csv file with the name my_solution.csv
my_solution4.to_csv("my_solution_4.csv", index_label = ["PassengerId"])


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()
t0 = time()
clf.fit(features_two, target)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(test_features)
print "predicting time:", round(time()-t0, 3), "s"
print pred
print ("Naive"), accuracy_score(target_test , pred)





# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution1 = pd.DataFrame(pred, PassengerId, columns = ["Survived"])

# Write your solution to a csv file with the name my_solution.csv
my_solution1.to_csv("my_solution_1.csv", index_label = ["PassengerId"])

#Neural Networks
from sklearn.neural_network import MLPClassifier
neural = MLPClassifier()
neural.fit(features_three,target)
pred_neural = neural.predict(test_features_three)
print ("Neural networks"), accuracy_score(target_test,pred_neural)

PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution_neural = pd.DataFrame(pred_neural, PassengerId, columns = ["Survived"])

# Write your solution to a csv file with the name my_solution.csv
my_solution_neural.to_csv("my_solution_neural.csv", index_label = ["PassengerId"])



