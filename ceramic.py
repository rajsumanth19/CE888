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
import seaborn as sns

df=pd.read_csv('Chemical Composion of Ceramic.csv')

print("Top 5 rows",df.head())

print("Column names",df.columns)

print("Descriptive statistics")
print(df.describe())


print(df.dtypes)

data = df.copy()

df.Part=df.Part.astype(str)

part_val = df.Part.value_counts()

df.isnull().sum()

df.Part.value_counts().plot(kind="bar")

# Encode all the categorical columns
for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes


         
# Define features and target varibles in the dfste
x = df.loc[:, df.columns != 'Part']
y= df['Part']

data.head()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(x)

x_train,x_test,y_train,y_test=tts(scaled,y,test_size=0.80,random_state=0)

forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(x_train,y_train)

y_pred_test = forest.predict(x_test)
y_pred_train = forest.predict(x_train)

evaluate(forest, x_train,x_test,y_train,y_test)
print(classification_report(y_test, y_pred_test, target_names=part_val.keys()))
print(classification_report(y_train, y_pred_train, target_names=part_val.keys()))

y_prob_test = list(forest.predict_proba(x_test))
y_prob_train = list(forest.predict_proba(x_train))

y_prob_train.extend(y_prob_test)

prob_class = list(part_val.keys())

new_class = []
for prob in y_prob_train:
    str1 = str(round(prob[0],1))+'-'+prob_class[0]+"-"+str(round(prob[1],1))+"-"+prob_class[1]
    new_class.append(str1)

data['new_class'] = new_class

data.head()

data.drop('Part',axis=1,inplace=True)

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
              ['RandomForest', 1, 1, 1, 1],
              ['DecisionTree', 0.06, 0.14 ,0.07, 0.17],
              ['DecisionTree after hyperparamter tunning', 0.05, 0.14 ,0.07, 0.32]
]
print(tabulate.tabulate(conclusion, tablefmt='fancy_grid'))
