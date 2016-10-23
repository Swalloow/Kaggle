
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
import gzip, csv

train = pd.read_csv("../input/train.csv")
print("get datasets")


# In[2]:

# Initailize datasets
train = pd.concat([train, pd.get_dummies(train["DayOfWeek"], prefix='Day')], axis=1)
train = pd.concat([train, pd.get_dummies(train['PdDistrict'], prefix='PdDistrict')], axis=1)

train.head()
predictors = train.columns.values[9:]


# In[4]:

x = train[predictors]
y = train["Category"]

train_data = train.values
train = DecisionTreeClassifier(max_depth=6)
train.fit(x[0:,2:], y[0:,0])


# In[ ]:

clf = tree.DecisionTreeClassifier()
iris = load_iris()

clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf, out_file='tree.dot')


# In[6]:

submit_data = pd.read_csv("../input/sampleSubmission.csv")
submit_data.head()

