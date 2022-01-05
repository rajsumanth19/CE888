import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from functions import *
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Reading Dataset
df=pd.read_csv('risk_factors_cervical_cancer.csv')

print("Top 5 rows")
print(df.head()) #show top 5 rows

print("Shape of dataset",df.shape)

df['Biopsy'].value_counts().plot(kind='bar')

data = df.copy()

df.Biopsy=df.Biopsy.astype(str)


print("Data types",df.dtypes)

part_val = df.Biopsy.value_counts()

print("Checking for NUll values",df.isnull().sum())


## Converting the object data type colum to categorical
for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes


# seprating independent and dependent variables
features = df.loc[:, df.columns != 'Biopsy']
target= df['Biopsy']


#spliting the dataset in train and test and we are dividing into 8:2 ratio
X_train, X_test, y_train, y_test= train_test_split(features,target,test_size=0.2, random_state=42)


#building Randomforest classifier
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train,y_train)

# Predicting the on test data
ynew = forest.predict(X_test)

#predicting the probability on the test data
forest.predict_proba(X_test)

y_pred_test = forest.predict(X_test)
y_pred_train = forest.predict(X_train)

#Evaluation of randomforest model on train and test data
evaluate(forest,X_train,X_test,y_train,y_test)


print(classification_report(y_test, y_pred_test))
print(classification_report(y_train, y_pred_train))

y_prob_test = list(forest.predict_proba(X_test))
y_prob_train = list(forest.predict_proba(X_train))

# Creating New column 
y_prob_train.extend(y_prob_test)

prob_class = list(part_val.keys())

new_class = []
for prob in y_prob_train:
    str1 = str(round(prob[0],1))+'-'+prob_class[0]+"-"+str(round(prob[1],1))+"-"+prob_class[1]
    new_class.append(str1)

data['new_class'] = new_class

data.head()

data.drop('Biopsy',axis=1,inplace=True)

for col_name in data.columns:
    if(data[col_name].dtype == 'object'):
        data[col_name]= data[col_name].astype('category')
        data[col_name] = data[col_name].cat.codes

data.head()

x = data.loc[:, data.columns != 'new_class']
y= data['new_class']


scaler = MinMaxScaler()
scaled = scaler.fit_transform(x)


x_train,x_test,y_train,y_test=tts(scaled,y,test_size=0.80,random_state=0)

#Building Decision tree model

dtree = DecisionTreeClassifier(max_depth = 2)
dtree.fit(x_train, y_train)


test_pred = dtree.predict(x_test)
cm = confusion_matrix(y_test, test_pred)

cm

print(classification_report(y_test, test_pred))

params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

#Hypertunning
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=dtree, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(x_train, y_train)

grid_search.best_estimator_

#Evaluate the best model 
evaluate(grid_search.best_estimator_,x_train,x_test,y_train,y_test)

import tabulate
conclusion = [['Model', 'Precision','Recall', 'F1score', 'Accuracy'],
              ['RandomForest', 0.76, 0.71, 0.73, 0.94],
              ['DecisionTree', 0.10, 0.10 ,0.10, 0.87],
              ['DecisionTree after hyperparamter tunning', 0.09, 0.1 ,0.09, 0.88]
]
print(tabulate.tabulate(conclusion, tablefmt='fancy_grid'))

