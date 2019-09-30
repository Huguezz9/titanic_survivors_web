import pandas as pd

# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

import matplotlib.pyplot as plt
# %matplotlib inline
# it will make plot outputs appear and they will be stored in the notebook
import seaborn as sns
sns.set()
# setting seaborn default for plots

def bar_chart(feature):
    # var.value_counts() returns a series containing counts of unique values.
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    # pandas.DataFrame() builds a table with the given values
    df = pd.DataFrame([survived, dead])
    # pandas.index stablish what the headers of the table will be
    df.index = ['Survived', 'Dead']
    # pandas.plot() shows a grafic made of bars in this case ('bar').
    df.plot(kind='bar', stacked=True, figsize=(10,5))

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train_test_data = [train, test]

    # For loop that checks each row of the combined dataset and extracts the title by taking what has a space before it and anything
    # until it sees a point.
    for dataset in train_test_data:
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                    "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Dona": 3, "Ms": 3, "Sir": 3, "Capt": 3,
                    "Don": 3, "Countess": 3, "Jonkheer": 3, "Mlle": 3, "Mme": 3, "Lady": 3, "Major": 3, "": 3}

    # pandas.map substitutes each value of the series by the value selected, in this case we defined that above.
    for dataset in train_test_data:
        dataset['Title'] = dataset['Title'].map(title_mapping)

    # pandas.drop deletes pieces of code by specifying label names and axis.
    train.drop('Name', axis=1, inplace=True)
    test.drop('Name', axis=1, inplace=True)

    sex_mapping = {"male": 0, "female": 1}

    for dataset in train_test_data:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping)

    # filling missing age with the median age for each title.
    train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
    test['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)

    train.groupby('Title')['Age'].transform('median')


    # pandas.loc locates the range selected and in this case we change it to an specific value depending on that range.
    for dataset in train_test_data:
        dataset.loc[dataset['Age'] <= 16, 'Age' ] = 0,
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age' ] = 1,
        dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age' ] = 2,
        dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age' ] = 3,
        dataset.loc[ dataset['Age'] > 62, 'Age' ] = 4

    Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
    Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
    Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    # mapping the 'Embarked'
    embarked_mapping = {'S': 0, 'C': 1, 'Q':2}

    for dataset in train_test_data:
        dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

    # filling missing fare
    train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    test['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace=True)

    for dataset in train_test_data:
        dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
        dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
        dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
        dataset.loc[dataset['Fare'] > 100, 'Fare' ] = 3

    for dataset in train_test_data:
        dataset['Cabin'] = dataset['Cabin'].str[:1]

    cabin_mapping = {'A': 0, 'B': 0.4, 'C': 0.8, 'D': 1.2, 'E':1.6, 'F': 2, 'G': 2.4, 'T': 2.8}
    for dataset in train_test_data:
        dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

    # filling na cabins
    train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
    test['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

    # Combine the passenger with the number of Sibsp and Parch
    train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
    test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

    features_drop = ['Ticket', 'SibSp', 'Parch', 'Title']
    train = train.drop(features_drop, axis=1)
    test = test.drop(features_drop, axis=1)
    train = train.drop(['PassengerId'], axis=1)

    # print(test.head())

    family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
    for dataset in train_test_data:
        dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


    train_data = train.drop('Survived', axis=1)
    target = train['Survived']


    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = SVC(gamma='auto')
    scoring = 'accuracy'
    score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

    clf = SVC(gamma='auto')
    clf.fit(train_data, target)

    test_data = test.drop("PassengerId", axis=1).copy()
    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
        "PassengerID": test["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('submission.csv', index=False)

    submission = pd.read_csv('submission.csv')

    last_value = submission['Survived'].values[-1]
    if last_value == 1:
        return True
    else:
        return False
main()