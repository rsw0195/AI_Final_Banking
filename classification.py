# -*- coding: utf-8 -*-
"""
Created on Thu Apr 05 11:40:27 2018

@author: Rebecca
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import time

# Data Clean-Up
data = pd.read_csv("bank-additional-full.csv", 
                   delimiter = ';')
print(data.shape)

data = data[data.job != 'unknown']
data = data[data.marital != 'unknown']
data = data[data.education != 'unknown']
data = data[data.default != 'unknown']
data = data[data.housing != 'unknown']
data = data[data.loan != 'unknown']
data = data[data.contact != 'unknown']

data['variation'] = data['emp.var.rate']
data['price'] = data['cons.price.idx']
data['confidence'] = data['cons.conf.idx']
data['employees'] = data['nr.employed']

def to_int(data, new_col, current_col):
    data[new_col] = data[current_col].apply(lambda x: 0 if x=='no' else 1)
    return data[new_col].value_counts()

to_int(data, "def_int", "default")
to_int(data, "house_int", 'housing')
to_int(data, 'loan_int', 'loan')
to_int(data, 'response', 'y')

date_fix = [data]
for column in date_fix:
    column.loc[column['month'] == 'jan', 'month_int'] = 1
    column.loc[column['month'] == 'feb', 'month_int'] = 2
    column.loc[column['month'] == 'mar', 'month_int'] = 3
    column.loc[column['month'] == 'apr', 'month_int'] = 4
    column.loc[column['month'] == 'may', 'month_int'] = 5
    column.loc[column['month'] == 'jun', 'month_int'] = 6
    column.loc[column['month'] == 'jul', 'month_int'] = 7
    column.loc[column['month'] == 'aug', 'month_int'] = 8
    column.loc[column['month'] == 'sep', 'month_int'] = 9
    column.loc[column['month'] == 'oct', 'month_int'] = 10
    column.loc[column['month'] == 'nov', 'month_int'] = 11
    column.loc[column['month'] == 'dec', 'month_int'] = 12
    column.loc[column['day_of_week'] == 'mon', 'day_int'] = 1
    column.loc[column['day_of_week'] == 'tue', 'day_int'] = 2
    column.loc[column['day_of_week'] == 'wed', 'day_int'] = 3
    column.loc[column['day_of_week'] == 'thu', 'day_int'] = 4
    column.loc[column['day_of_week'] == 'fri', 'day_int'] = 5
    column.loc[column['job'] == 'admin.', 'emp'] = 10
    column.loc[column['job'] == 'self-employed', 'emp'] = 4
    column.loc[column['job'] == 'blue-collar', 'emp'] = 9
    column.loc[column['job'] == 'entrepreneur', 'emp'] = 8
    column.loc[column['job'] == 'housemaid', 'emp'] = 7
    column.loc[column['job'] == 'management', 'emp'] = 6
    column.loc[column['job'] == 'retired', 'emp'] = 3
    column.loc[column['job'] == 'services', 'emp'] = 5
    column.loc[column['job'] == 'student', 'emp'] = 2
    column.loc[column['job'] == 'technician', 'emp'] = 11
    column.loc[column['job'] == 'unemployed', 'emp'] = 1
    column.loc[column['marital'] == 'married', 'mar'] = 2
    column.loc[column['marital'] == 'divorced', 'mar'] = 3
    column.loc[column['marital'] == 'single', 'mar'] = 1
    column.loc[column['education'] == 'basic.4y', 'edu'] = 1
    column.loc[column['education'] == 'basic.6y', 'edu'] = 2
    column.loc[column['education'] == 'basic.9y', 'edu'] = 3
    column.loc[column['education'] == 'high.school', 'edu'] = 4
    column.loc[column['education'] == 'professional.course', 'edu'] = 5
    column.loc[column['education'] == 'university.degree', 'edu'] = 6
    column.loc[column['education'] == 'illiterate', 'edu'] = 7
    
data['month_int'] = data['month_int'].astype(np.int64)
data['day_int'] = data['day_int'].astype(np.int64)
data['emp'] = data['emp'].astype(np.int64)
data['mar'] = data['mar'].astype(np.int64)
data['edu'] = data['edu'].astype(np.int64)

data['prev'] = data['pdays'].apply(lambda x: 1 if x != 999 else (1 if x == 0 else 0))
data['cont'] = data['contact'].apply(lambda x: 1 if x == 'cellular' else 2)
    
data['outcome'] = data['poutcome'].apply(lambda x: 0 if x == 'failure' else (1 if x == 'success' else np.NaN))
data = data.dropna()
data['outcome'] = data['outcome'].astype(np.int64)

data.drop(['job', 'marital', 'education', 'contact', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed', 'poutcome', 'default', 'housing', 'loan', 'y', 'duration', 'day_of_week', 'month', 'pdays'], axis = 1, inplace = True)

# Create test/train sets
y = data['outcome']
x = data.iloc[:,0:19]

x_minmax = preprocessing.normalize(x, norm = 'l2')
x = preprocessing.scale(x_minmax)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)

'''
A reference from GitHub was used to help effectively compare the various models - 
https://www.kaggle.com/janiobachmann/bank-classifying-term-deposit-subscriptions
'''
class_dict = {'Logistic Regression': LogisticRegression(),
              'KNN': KNeighborsClassifier(),
              'Linear SVM': SVC(),
              'Gradient Boosting': GradientBoostingClassifier(),
              'Decision Tree': tree.DecisionTreeClassifier(),
              'Random Forest': RandomForestClassifier(n_estimators = 100),
              'Neural Net': MLPClassifier(alpha = 0.1),
              'Naive Bayes': GaussianNB()}

no_class = len(class_dict.keys())

    

def batch_class(c_train, y_train, verbose = True):
    df_result = pd.DataFrame(data = np.zeros(shape = (no_class, 8)), columns = ['classifier', 'trainScore', 'testScore', 'truePos', 'trueNeg', 'falsePos', 'falseNeg', 'time'])
    count = 0
    for key, classifier in class_dict.items():
        t_start = time.clock()
        classifier.fit(x_train, y_train)
        y_hat = classifier.predict(x_test)
        t_end = time.clock()
        t_diff = t_end - t_start
        train_score = classifier.score(x_train, y_train)
        tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
        df_result.loc[count, 'classifier'] = key
        df_result.loc[count, 'trainScore'] = train_score*100
        df_result.loc[count, 'time'] = t_diff
        df_result.loc[count, 'testScore'] = accuracy_score(y_hat, y_test)*100
        df_result.loc[count, 'truePos'] = float(tp)/(tp + fn)*100
        df_result.loc[count, 'trueNeg'] = float(tn)/(tn+fp)*100
        df_result.loc[count, 'falsePos'] = float(fp)/(fp+tn)*100
        df_result.loc[count, 'falseNeg'] = float(fn)/(tp+fn)*100
        if verbose:
            print( "trained {c} in {f:.2f} s".format(c = key, f = t_diff))
        count += 1
    return df_result

df_result = batch_class(x_train, y_train)
df_final = df_result.sort_values(by = 'testScore', ascending = False).copy()

writer = pd.ExcelWriter('finaltable1.xlsx')
df_final.to_excel(writer, 'Sheet1')
writer.save()

# Print table of Model Comparisons
print('_______________________________________')
print(df_final)
print('** Score Values given as percents **')

# Print Best Model Results
print('_______________________________________')
print("Best Model: \n")
print(np.transpose(df_final.head(1)))

# Print Neural Network Model Results
print('_________________________________________')
print('Classification Neural Network: \n')
print(df_final.loc[1,])
