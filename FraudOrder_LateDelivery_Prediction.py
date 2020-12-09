#!/usr/bin/env python
# coding: utf-8

#  ## fraud orders, late delivery prediction
#  - models： LR，GuassianNB, LinearSVC, KNeighborsClassifier, LinearDisceiminantAnalysis, DecisionTreeClassifier, RandomForestClassifier, XGBCLassifier

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

import pickle
with open('data/SupplyChain.pkl', 'rb') as file:
    train = pickle.load(file)
print(train.shape)
train.head()

# Create Target label
train['fraud'] = np.where(train['Order Status']=='SUSPECTED_FRAUD',1,0)
train['late_delivery'] = np.where(train['Delivery Status']=='Late delivery',1,0) 
# train.info()

categorical_cols = train.select_dtypes(include='object').columns.to_list()
categorical_cols

# drop unnecessary categorical columns
drop_cols = ['Customer Email','Customer Fname', 'Customer Lname','Customer Password',
             'Product Image','Product Status','Late_delivery_risk']
train.drop(drop_cols, axis=1, inplace=True)
# train.shape
# train.info()

# new categorical columns
categorical_cols = train.select_dtypes(include='object').columns
# categorical_cols

# check nulls before label encoder
null_index = train[train['Customer Full Name'].isnull()]
train.dropna(subset=['Customer Full Name'],inplace=True)

# label encoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = label_encoder.fit_transform(train[col])
train[categorical_cols]
# train.info()

# deal with time related features
train['shipping date (DateOrders)'] = pd.to_datetime(train['shipping date (DateOrders)'])
train['Prepare Time'] = train['shipping date (DateOrders)']-train['order date (DateOrders)']
# train['Prepare Time'].apply(lambda x: x.days)
# shipping date will be fropped
train.drop(['order date (DateOrders)','Order Month Year','shipping date (DateOrders)','Prepare Time'],axis=1, inplace=True)


# numeric features
categorical_cols = categorical_cols.to_list()
categorical_cols.remove('shipping date (DateOrders)')
numerical_cols = train.columns.to_list()
for col in categorical_cols:
    numerical_cols.remove(col)
# train[numerical_cols]


# ### Fraud Orders prediction

# create data
fraud_x = train.loc[:,train.columns!='fraud']
fraud_y = train['fraud']

fraud_x.drop('Order Status',axis = 1, inplace = True)

# split train and test data
from sklearn.model_selection import train_test_split
fraud_x_train,fraud_x_test, fraud_y_train, fraud_y_test = train_test_split(fraud_x,fraud_y,test_size = 0.2)


# standardization
sscaler = StandardScaler()
fraud_x_train = sscaler.fit_transform(fraud_x_train)
fraud_x_test = sscaler.transform(fraud_x_test)

# make a function for modeling
from sklearn.metrics import accuracy_score, recall_score,roc_auc_score,confusion_matrix,f1_score
def model_stats(model, x_train, x_test,y_train,y_test, name='fraud'):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy  = accuracy_score(y_pred,y_test)
    recall = recall_score(y_pred,y_test)
    f1 = f1_score(y_pred,y_test)
    auc = roc_auc_score(y_pred, y_test)
    confusion = confusion_matrix(y_pred, y_test)

    print('model is ：', model)
    print('accuracy of {}：{}%'.format(name, accuracy*100))
    print('recall of {}：{}%'.format(name, recall*100))
    print('F1 score of {}：{}%'.format(name, f1*100))
    print('AUC of{}：{}%'.format(name, auc*100))
    print('confusion matrix of {}：\n{}'.format(name, confusion))

    return accuracy, recall, f1

# use model in fit

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from xgboost import XGBClassifier

accuracy, recall, f1 = [],[],[]


# Logistic Regression
lr_fraud = LogisticRegression()
result = model_stats(lr_fraud, fraud_x_train,fraud_x_test,fraud_y_train,fraud_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])

# linear Support Vector Classifier
svc_fraud = LinearSVC()
result = model_stats(svc_fraud, fraud_x_train,fraud_x_test,fraud_y_train,fraud_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# Gaussian Naive Bayes
nb_fraud = GaussianNB()
result = model_stats(nb_fraud, fraud_x_train,fraud_x_test,fraud_y_train,fraud_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# K Nearest Neighbors
knn_fraud = KNeighborsClassifier()
result = model_stats(knn_fraud, fraud_x_train,fraud_x_test,fraud_y_train,fraud_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# Decision Tree
dt_fraud = DecisionTreeClassifier()
result = model_stats(dt_fraud, fraud_x_train,fraud_x_test,fraud_y_train,fraud_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# Random Forest
rf_fraud = RandomForestClassifier()
result = model_stats(rf_fraud, fraud_x_train,fraud_x_test,fraud_y_train,fraud_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# Linear Discriminant Analysis
lda_fraud = LinearDiscriminantAnalysis()
result = model_stats(lda_fraud, fraud_x_train,fraud_x_test,fraud_y_train,fraud_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# In[27]:


# Result table
models = ['LogisticRegression','LinearSVC','GaussianNB','KNeighborsClassifier','DecisionTree','RandomForest','LDA']
re = pd.DataFrame({'Models':models,'Accuracy':accuracy,'Recall':recall,'F1 Score':f1})
re


#  ## Late delivery Prediction


# Get data
late_x = train.loc[:,train.columns!='late_delivery']
late_y = train['late_delivery']

late_x.drop('Delivery Status',axis = 1, inplace = True)

# split data
late_x_train,late_x_test, late_y_train, late_y_test = train_test_split(late_x,late_y,test_size = 0.2)

# standardization
sscaler = StandardScaler()
fraud_x_train = sscaler.fit_transform(late_x_train)
fraud_x_test = sscaler.fit(late_x_test)

accuracy,recall, f1 = [],[],[]


# Logistic Regression
lr_late = LogisticRegression()
result = model_stats(lr_late, late_x_train,late_x_test,late_y_train,late_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# Linear SVC
svc_late = LinearSVC()
result = model_stats(svc_late, late_x_train,late_x_test,late_y_train,late_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# NB
nb_late = GaussianNB()
result = model_stats(nb_late, late_x_train,late_x_test,late_y_train,late_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# knn
knn_late = KNeighborsClassifier()
result = model_stats(knn_late, late_x_train,late_x_test,late_y_train,late_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# Decision Tree
dt_late = DecisionTreeClassifier()
result = model_stats(dt_late, late_x_train,late_x_test,late_y_train,late_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# Random Forest
rf_late = RandomForestClassifier()
result = model_stats(rf_late, late_x_train,late_x_test,late_y_train,late_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# Linear Discriminant Analysis
lda_late = LinearDiscriminantAnalysis()
result = model_stats(lda_late, late_x_train,late_x_test,late_y_train,late_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])


# Result Table
models = ['LogisticRegression','LinearSVC','GaussianNB','KNeighborsClassifier','DecisionTree','RandomForest','LDA']
re = pd.DataFrame({'Models':models,'Accuracy':accuracy,'Recall':recall,'F1 Score':f1})
# re


# Check label leakage of late delivery
import seaborn as sns
plt.figure(figsize=(50,50))
sns.heatmap(train.corr(),annot=True,cmap = 'coolwarm')
plt.show()


# train['Product Status'].value_counts()
# len(train[train['Late_delivery_risk']==train['late_delivery']])==len(train)


# #### drop 'Late_delivery_risk'
