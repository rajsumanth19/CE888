import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tabulate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from functions import *

df=pd.read_excel('Immunotherapy.xlsx')

print("Top 5 rows",df.head())

print("Column names",df.columns)

print("Descriptive statistics")
print(df.describe())


print(df.dtypes)

data = df.copy()

df.Result_of_Treatment=df.Result_of_Treatment.astype(str)

part_val = df.Result_of_Treatment.value_counts()

df.isnull().sum()


for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes

features = df.loc[:, df.columns != 'Result_of_Treatment']
target= df['Result_of_Treatment']

X_train, X_test, y_train, y_test= train_test_split(features,target,test_size=0.2, random_state=42)


forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train,y_train)

ynew = forest.predict(X_test)

forest.predict_proba(X_test)

y_pred_test = forest.predict(X_test)
y_pred_train = forest.predict(X_train)

evaluate(forest,X_train,X_test,y_train,y_test)
print(classification_report(y_test, y_pred_test))
print(classification_report(y_train, y_pred_train))

y_prob_test = list(forest.predict_proba(X_test))
y_prob_train = list(forest.predict_proba(X_train))

y_prob_train.extend(y_prob_test)

prob_class = list(part_val.keys())

new_class = []
for prob in y_prob_train:
    str1 = str(round(prob[0],1))+'-'+prob_class[0]+"-"+str(round(prob[1],1))+"-"+prob_class[1]
    new_class.append(str1)

data['new_class'] = new_class

data.head()

data.drop('Result_of_Treatment',axis=1,inplace=True)

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


grid_search = GridSearchCV(estimator=dtree, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(x_train, y_train)

grid_search.best_estimator_

evaluate(grid_search.best_estimator_,x_train,x_test,y_train,y_test)

conclusion = [['Model', 'Precision','Recall', 'F1score', 'Accuracy'],
              ['RandomForest', 0.88, 0.60, 0.60, 0.78],
              ['DecisionTree', 0.09, 0.08 ,0.08, 0.24],
              ['DecisionTree after hyperparamter tunning', 0.1, 0.13 ,0.12, 0.42]
]
print(tabulate.tabulate(conclusion, tablefmt='fancy_grid'))
