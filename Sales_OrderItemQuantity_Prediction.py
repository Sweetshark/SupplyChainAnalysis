#!/usr/bin/env python
# coding: utf-8

#  ##  sales/orders prediction
#  - model：LinearRegression, Lasso, Ridge, DecisionTreeRegressor, XGBRegressor, LGBMRegressor, RandomForestRegressor

# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# In[3]:


import pickle
with open('data/SupplyChain.pkl', 'rb') as file:
    train = pickle.load(file)
print(train.shape)
train.head()

# makr sure the tragets
print(train['Sales'].value_counts())
print(train['Order Item Quantity'].value_counts())



# similar features, drop one
train['Category Name'].value_counts()
train['Product Category Id'].value_counts()
###
train['Days for shipment (scheduled)'].value_counts()
train['Shipping Mode'].value_counts()
###

#  drop unnecessary categorical columns
drop_cols = ['Customer Email','Customer Fname', 'Customer Lname','Customer Password',
             'Product Image','Product Status','Late_delivery_risk','Order Item Total',
            'Latitude','Longitude','Product Price','Category Name','Shipping Mode']
train.drop(drop_cols, axis=1, inplace=True)


categorical_cols = train.select_dtypes(include='object').columns.to_list()
categorical_cols



# final categorical columns
categorical_cols = train.select_dtypes(include='object').columns
categorical_cols


# check nulls before label encoder
# null_index = train[train['Customer Full Name'].isnull()]
train.dropna(subset=['Customer Full Name'],inplace=True)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = label_encoder.fit_transform(train[col])
# train[categorical_cols]



# deal with time fatures
train['shipping date (DateOrders)'] = pd.to_datetime(train['shipping date (DateOrders)'])
train['Prepare Time'] = train['shipping date (DateOrders)']-train['order date (DateOrders)']
# train['Prepare Time'].apply(lambda x: x.days)
train.drop(['order date (DateOrders)','Order Month Year','shipping date (DateOrders)','Prepare Time'],axis=1, inplace=True)

# numeric features
categorical_cols = categorical_cols.to_list()
categorical_cols.remove('shipping date (DateOrders)')
numerical_cols = train.columns.to_list()
for col in categorical_cols:
    numerical_cols.remove(col)
# train[numerical_cols]


# check values for each column
for col in train.columns:
    if len(train[col].value_counts())<2:
        print(col)



# ### Sales prediction


# get data
sales_x = train.iloc[:,train.columns!='Sales']
sales_y = train['Sales']

# split
from sklearn.model_selection import train_test_split
sales_x_train,sales_x_test, sales_y_train, sales_y_test = train_test_split(sales_x,sales_y,test_size = 0.2)


# standardization
from sklearn.preprocessing import StandardScaler
sscaler = StandardScaler()
fraud_x_train = sscaler.fit_transform(sales_x_train)
fraud_x_test = sscaler.transform(sales_x_test)


# build a function
from sklearn.metrics import mean_absolute_error,mean_squared_error
def model_stats(model, x_train, x_test,y_train,y_test, name='Sales'):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    print('使用的模型是：',model)
    print('{}的mae：{}'.format(name,mae))
    print('{}的mse：{}'.format(name,mse))

    return mae, mse



# load modules
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm.sklearn import LGBMRegressor

mae,mse= [],[]



# Linear Regression
lr_sales = LinearRegression()
result = model_stats(lr_sales, sales_x_train,sales_x_test,sales_y_train,sales_y_test)
mae.append(result[0])
mse.append(result[1])

## ridge
ridge_sales = Ridge()
result = model_stats(ridge_sales, sales_x_train,sales_x_test,sales_y_train,sales_y_test)
mae.append(result[0])
mse.append(result[1])

## lasso
lasso_sales = Lasso()
result = model_stats(lasso_sales, sales_x_train,sales_x_test,sales_y_train,sales_y_test)
mae.append(result[0])
mse.append(result[1])


# svr ## not applicable
# svr_sales = SVR()
# result = model_stats(svr_sales, sales_x_train,sales_x_test,sales_y_train,sales_y_test)
# mae.append(result[0])
# mse.append(result[1])


# knn Regressor
knn_sales = KNeighborsRegressor()
result = model_stats(knn_sales, sales_x_train,sales_x_test,sales_y_train,sales_y_test)
mae.append(result[0])
mse.append(result[1])


# Decision Tree regressor
dt_sales = DecisionTreeRegressor()
result = model_stats(dt_sales, sales_x_train,sales_x_test,sales_y_train,sales_y_test)
mae.append(result[0])
mse.append(result[1])


# Random Forest Regressor
rf_sales = RandomForestRegressor()
result = model_stats(rf_sales, sales_x_train,sales_x_test,sales_y_train,sales_y_test)
mae.append(result[0])
mse.append(result[1])


# xgb Regressor
xgb_salse = XGBRegressor()
result = model_stats(xgb_salse,sales_x_train,sales_x_test,sales_y_train,sales_y_test)
mae.append(result[0])
mse.append(result[1])


# lightgbm Regressor
lgbm_salse = LGBMRegressor()
result = model_stats(lgbm_salse,sales_x_train,sales_x_test,sales_y_train,sales_y_test)
mae.append(result[0])
mse.append(result[1])


# show Result table
models = ['LinearRegression','Ridge','Lasso','KNNRegressor','DecisionTreeRegressor','RandomForest','XGB','LightGBM']
re = pd.DataFrame({'Models':models,'Mean Absolute Error':mae,'Mean Squared Error':mse})
# re


#  ## order item quantity prediction



# get data
order_x = train.loc[:,train.columns!='Order Item Quantity']
order_y = train['Order Item Quantity']

# split
order_x_train,order_x_test, order_y_train, order_y_test = train_test_split(order_x,order_y,test_size = 0.2)

# standardization
sscaler = StandardScaler()
fraud_x_train = sscaler.fit_transform(order_x_train)
fraud_x_test = sscaler.fit(order_x_test)

mae,mse = [],[]



# Linear Regression
lr_order = LinearRegression()
result = model_stats(lr_order,order_x_train,order_x_test, order_y_train, order_y_test)
mae.append(result[0])
mse.append(result[1])

## ridge
ridge_order = Ridge()
result = model_stats(ridge_order, order_x_train,order_x_test, order_y_train, order_y_test)
mae.append(result[0])
mse.append(result[1])

## lasso
lasso_order = Lasso()
result = model_stats(lasso_order,order_x_train,order_x_test, order_y_train, order_y_test)
mae.append(result[0])
mse.append(result[1])


# svr
#svr_order = SVR()
#result = model_stats(svr_order, order_x_train,order_x_test, order_y_train, order_y_test)
#mae.append(result[0])
#mse.append(result[1])


# knn
knn_order = KNeighborsRegressor()
result = model_stats(knn_order, order_x_train,order_x_test, order_y_train, order_y_test)
mae.append(result[0])
mse.append(result[1])


# Decision Tree
dt_order = DecisionTreeRegressor()
result = model_stats(dt_order, order_x_train,order_x_test, order_y_train, order_y_test)
mae.append(result[0])
mse.append(result[1])


# Random Forest
rf_order = RandomForestRegressor()
result = model_stats(rf_order, order_x_train,order_x_test, order_y_train, order_y_test)
mae.append(result[0])
mse.append(result[1])


# xgb
xgb_order = XGBRegressor()
result = model_stats(xgb_order,order_x_train,order_x_test, order_y_train, order_y_test)
mae.append(result[0])
mse.append(result[1])


# lightgbm
lgbm_order = LGBMRegressor()
result = model_stats(lgbm_order,order_x_train,order_x_test, order_y_train, order_y_test)
mae.append(result[0])
mse.append(result[1])


# result table
models = ['LinearRegression','Ridge','Lasso','KNNRegressor','DecisionTreeRegressor','RandomForest','XGB','LightGBM']
re = pd.DataFrame({'Models':models,'Mean Absolute Error':mae,'Mean Squared Error':mse})
# re


index_sorted = lr_order.coef_.argsort()
pd.DataFrame({'Feature':order_x.columns[index_sorted],'Coefficient':lr_order.coef_[index_sorted]})


## ：feature importance
importance_col = dt_order.feature_importances_.argsort()#从小到大排并输出index
#print(importance_col)
#print(dt_fraud.feature_importances_)
feat_importance = pd.DataFrame({'feature':order_x.columns[importance_col],'importance':dt_order.feature_importances_[importance_col]})
feat_importance


# ## Sales 和Product Name have large feature importance


train['Order Year'].value_counts()




