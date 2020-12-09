#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sc = pd.read_csv('data/SupplyChain.csv', encoding='unicode_escape')
# sc.head()
# sc.shape

# check the null values
null_value = sc.isnull().sum()
# print(null_value[null_value>0])

# Deal with null values
# 1.create new feature: full name
sc['Customer Full Name'] = sc['Customer Fname'] + sc['Customer Lname']
# 2.fill with 0
sc['Customer Zipcode'].value_counts()
sc['Customer Zipcode'].fillna(0, inplace=True)
# 3.drop
sc.drop(['Order Zipcode', 'Product Description'], axis=1, inplace=True)

# correlation analysis
plt.figure(figsize=(24, 18))
sns.heatmap(sc.corr(), annot=True, cmap='coolwarm')
plt.show()

# get the features of high relations and drop them
features = ['Category Id', 'Customer Zipcode', 'Order Customer Id', 'Department Id',
            'Sales per customer', 'Order Item Id', 'Benefit per order', 'Order Item Cardprod Id',
            'Product Card Id', 'Order Item Product Price']
sc.drop(features, axis=1, inplace=True)

# check correlation again
plt.figure(figsize=(24, 18))
sns.heatmap(sc.corr(), annot=True, cmap='coolwarm')
plt.show

# See sales from different perspectives
# group by market
# sc.Market.value_counts()
market = sc.groupby('Market')
market['Sales'].sum().sort_values(ascending=False).plot.bar(title='Sales in Different Market')

# group by region
region = sc.groupby('Order Region')
region['Sales'].sum().sort_values(ascending=False).plot.bar(title='Sales in Different Regions')

# group by product category
category = sc.groupby('Category Name')
category['Sales'].sum().sort_values(ascending=False).plot.bar(figsize=(12, 6), title='Sales in Different Categories')

# group by time
# generate year, month, weekday, hour, month year
time = pd.DatetimeIndex(sc['order date (DateOrders)'])
sc['Order Year'] = time.year
sc['Order Month'] = time.month
sc['Order Weekday'] = time.weekday
sc['Order Hour'] = time.hour
sc['Order Month Year'] = time.to_period('M')
sc[['Order Year', 'Order Month', 'Order Weekday', 'Order Hour', 'Order Month Year']]
# group by time to see average sales
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
order_year = sc.groupby('Order Year')
order_year['Sales'].mean().plot(title='Average Sales in Each Year')
plt.subplot(2, 2, 2)
order_month = sc.groupby('Order Month')
order_month['Sales'].mean().plot(title='Average Sales in Each Month')
plt.subplot(2, 2, 3)
order_weekday = sc.groupby('Order Weekday')
order_weekday['Sales'].mean().plot(title='Average Sales in Each Weekday')
plt.subplot(2, 2, 4)
order_hour = sc.groupby('Order Hour')
order_hour['Sales'].mean().plot(title='Average Sales in Each Hour')

# explore relationship between product price and sales
plt.scatter(x=sc['Product Price'], y=sc['Sales'])
plt.title('Relationship Between Product Price and Sales')

# FRM
import numpy as np
import datetime

# Calculate R_value，F_value,M_value, Group by Customer Id
# 1.look at the date
sc['order date (DateOrders)'] = pd.to_datetime(sc['order date (DateOrders)'])
sc['order date (DateOrders)'].max()
# Assume today is 2018-02-01

current = datetime.datetime(2018, 2, 1)
# print(current)

# 2.Define recency as the difference between the date of  last order and today: R_value
# Calculate the number orders and the total sales as Frequency and Monetary Value: M_Value, R_Value
customer_seg = sc.groupby('Customer Id').agg({'order date (DateOrders)': lambda x: (current - x.max()).days,
                                              'Order Id': lambda x: len(x), 'Sales': lambda x: x.sum()})
customer_seg.rename(columns={'order date (DateOrders)': 'R_value',
                             'Order Id': 'F_value', 'Sales': 'M_value'}, inplace=True)

# 3.Divide RFM into four groups in terms of value
# get the thresh
quantiles = customer_seg.quantile(q=[0.25, 0.5, 0.75])
# quantiles

# Deal with R_value: the smaller, the better to get R_score.
def R_score(a, b, c):
    if a <= c[b][0.25]:
        return 4
    if a <= c[b][0.5]:
        return 3
    if a <= c[b][0.75]:
        return 2
    return 1


# Deal with F_value and M_value: the larger, the better to get F_score and M_score.
def FM_score(a, b, c):
    if a <= c[b][0.25]:
        return 1
    if a <= c[b][0.5]:
        return 2
    if a <= c[b][0.75]:
        return 3
    return 4


customer_seg['R_score'] = customer_seg['R_value'].apply(R_score, args=('R_value', quantiles))
customer_seg['F_score'] = customer_seg['F_value'].apply(FM_score, args=('F_value', quantiles))
customer_seg['M_score'] = customer_seg['M_value'].apply(FM_score, args=('M_value', quantiles))


# do customer segementation
def RFM_User(df):
    if df['M_score'] > 2 and df['R_score'] > 2 and df['F_score'] > 2:
        return 'Importantly Valuable'
    if df['M_score'] > 2 and df['R_score'] > 2 and df['F_score'] <= 2:
        return 'Importantly Developing'
    if df['M_score'] > 2 and df['R_score'] <= 2 and df['F_score'] > 2:
        return 'Importantly Retaining'
    if df['M_score'] > 2 and df['R_score'] <= 2 and df['F_score'] <= 2:
        return 'Importantly Detaining'

    if df['M_score'] <= 2 and df['R_score'] > 2 and df['F_score'] > 2:
        return 'Normally Valuable'
    if df['M_score'] <= 2 and df['R_score'] > 2 and df['F_score'] <= 2:
        return 'Normally Developing'
    if df['M_score'] <= 2 and df['R_score'] <= 2 and df['F_score'] > 2:
        return 'Normally Retaining'
    if df['M_score'] <= 2 and df['R_score'] <= 2 and df['F_score'] <= 2:
        return 'Normally Detaining'

customer_seg['Customer Segmentation'] = customer_seg.apply(RFM_User, axis=1)
customer_seg.head(10)


##
# payment
sc.Type.value_counts()

pay_type1 = sc[sc.Type == 'DEBIT']
pay_type2 = sc[sc.Type == 'TRANSFER']
pay_type3 = sc[sc.Type == 'PAYMENT']
pay_type4 = sc[sc.Type == 'CASH']

# 1. group by region
count1 = pay_type1['Order Region'].value_counts()
count2 = pay_type2['Order Region'].value_counts()
count3 = pay_type3['Order Region'].value_counts()
count4 = pay_type4['Order Region'].value_counts()

region_num = sc['Order Region'].nunique()
fig, ax = plt.subplots(figsize=(20, 10))
index = np.arange(region_num)
regions = sc['Order Region'].to_list()

bar_width = 0.2
plt.bar(index, count1, bar_width, color='b', label='DEBIT')
plt.bar(index + bar_width, count2, bar_width, color='r', label='TRANSFER')
plt.bar(index + bar_width * 2, count3, bar_width, color='y', label='PAYMENT')
plt.bar(index + bar_width * 3, count4, bar_width, color='g', label='CASH')
plt.xlabel('Order Region')
plt.ylabel('Numer of Payments')
plt.title('Types of Payments in Different Regions')
plt.legend()
plt.xticks(index + bar_width, regions, rotation='vertical')
plt.show()

# ### Conclusion：
# - DEBIT is the most popular payment
# - CASH is least popular.

# explore negative order, fraud order and late delivery.

# explore profits
negative = sc[sc['Order Profit Per Order'] < 0]
# 1. by category
negative['Category Name'].value_counts().nlargest(10).plot.bar(figsize=(8, 6),
                                                               title='Top10 Product Categories with Negative Profits')
# 2. by region
negative['Order Region'].value_counts().nlargest(10).plot.bar(figsize=(8, 6),
                                                              title='Top10 Regions with Negative Profits')
# Total loss
print('Total Negative profits: ', negative['Order Profit Per Order'].sum())

# explore fraud orders
sc['Order Status'].value_counts()
sc[sc['Order Status'] == 'SUSPECTED_FRAUD']['Type'].value_counts()

# - TRANSFER is easily to make fraud order take place

# 1. by region
suspected_fraud = sc[sc['Order Status'] == 'SUSPECTED_FRAUD']
suspected_fraud['Order Region'].value_counts().plot.bar(figsize=(8, 6), title='Top10 Regions of Suspected Fraud')
plt.ylabel('Number of Suspected Fraud Order')
plt.xlabel('Product Categories')
plt.show()

# 2. by category
suspected_fraud['Category Name'].value_counts().nlargest(10).plot.bar(figsize=(12, 6),
                                                                      title='Top10 Categories of Suspected Fraud',
                                                                      label='All regions')
suspected_fraud[suspected_fraud['Order Region'] == 'Western Europe']['Category Name'].value_counts().nlargest(
    10).plot.bar(figsize=(12, 6), color='r', label='Western Europe')
plt.legend()
plt.ylabel('Number of Suspected Fraud')
plt.xlabel('Order Regions')
plt.show()

# 3. by customer
sc[sc['Order Status'] == 'SUSPECTED_FRAUD']['Customer Full Name'].value_counts()

suspected_fraud['Customer Full Name'].value_counts().nlargest(10).plot.bar(figsize=(8, 4),
                                                                           title='Top 10 Customers of Suspected Fraud Orders')
plt.xlabel('Customer Full Name')
plt.ylabel('Number of Suspected Fraud Orders')
plt.show()

# MarySmith is suspect
total = sc[sc['Customer Full Name'] == 'MarySmith']['Sales'].sum()
fraud = suspected_fraud[suspected_fraud['Customer Full Name'] == 'MarySmith']['Sales'].sum()
print('MarySmith total sales：', total, 'MarySmith fraud order sales：', fraud)


# explore late delivery
sc['Delivery Status'].value_counts()
late = sc[sc['Delivery Status'] == 'Late delivery']

# 1. by region
late['Order Region'].value_counts().nlargest(10).plot.bar(figsize=(8, 4), title='Top10 Regions of Late Delivery')
plt.xlabel('Regions')
plt.ylabel('Number of Late Delivery Orders')
plt.show()

# 2. by category
late['Category Name'].value_counts().nlargest(10).plot.bar(figsize=(8, 4), title='Top10 Categories of Late Delivery')
plt.xlabel('Product Categories')
plt.ylabel('Number of Late Delivery Orders')
plt.show()

# save in the pickle for convenience
import pickle

with open('data/SupplyChain.pkl', 'wb') as file:
    pickle.dump(sc, file)
