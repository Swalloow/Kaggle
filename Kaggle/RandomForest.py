
# coding: utf-8

# In[124]:

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation


# In[129]:

titanic = pd.read_csv("../input/train.csv")
titanic.head()


# In[130]:

Title_Dictionary = {
                    "Capt":       "0",
                    "Col":        "0",
                    "Major":      "0",
                    "Jonkheer":   "1",
                    "Don":        "1",
                    "Sir" :       "1",
                    "Dr":         "0",
                    "Rev":        "0",
                    "the Countess":"1",
                    "Dona":       "1",
                    "Mme":        "2",
                    "Mlle":       "3",
                    "Ms":         "2",
                    "Mr" :        "4",
                    "Mrs" :       "2",
                    "Miss" :      "3",
                    "Master" :    "6",
                    "Lady" :      "1"
}


# In[131]:

# Initialize titanic dataset
def Init_Data(titanic):
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mean())
    
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 1
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 0
    
    titanic["Embarked"]=titanic["Embarked"].fillna("S")
    titanic.loc[titanic["Embarked"]=="S","Embarked"] = 0
    titanic.loc[titanic["Embarked"]=="C","Embarked"] = 1
    titanic.loc[titanic["Embarked"]=="Q","Embarked"] = 2
    
    titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
    
    titanic['Title'] = titanic['Name'].apply(lambda x: Title_Dictionary[x.split(',')[1].split('.')[0].strip()])

    return titanic


# In[132]:

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize", "Title"]
titanic = Init_Data(titanic)
x = titanic[predictors]
y = titanic["Survived"]
titanic.head()


# In[133]:

titanic_test = pd.read_csv("../input/test.csv", header=0)
titanic_test = Init_Data(titanic_test)
titanic_test.head()


# In[134]:

predictors_test = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title"]
if len(titanic_test.Fare[ titanic_test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = titanic_test[ titanic_test.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        titanic_test.loc[ (titanic_test.Fare.isnull()) & (titanic_test.Pclass == f+1 ), 'Fare'] = median_fare[f]
x_test = titanic_test[predictors_test]


# In[135]:

clf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
forst = clf.fit(x,y)
print ("Predicting...")
output = forst.predict(x_test).astype(int)


# In[136]:

submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'],'Survived': output})
submission.to_csv("RandomForest2.csv", index=False)

