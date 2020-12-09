# Supply Chain Analysis

## Introduction 
- Fields Decription
![| Fields | Description  |
|--|--|
| Type | Type of transaction made |](https://img-blog.csdnimg.cn/20201207211937245.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N3ZWV0c2hhcms=,size_30,color_FFFFFF,t_70)

This dataset is downloaded from Internet. I can assume it is generated from an e-commerce company. Private infomation have already been preprocessed. After a glance of the data, I found all features could be divided into three groups. Customer related, Order related and Product related. Some variables have similar meanings, so correlation analysis will be included in the following EDA step.

Then I asked myself: what can I do with this dataset? 

Then here came my first goal. Costomer is always the main focus of e-commerce companies. So, I used a FRM model to make customer segmentation in terms of customer value.

## EDA 

- For this part, I will work on data frame and do information visualization. So, I load these three modules.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
-  Load data and see the structure.
```python
print(sc.shape)
sc.head()
```

    (180519, 53)
    
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Days for shipping (real)</th>
      <th>Days for shipment (scheduled)</th>
      <th>Benefit per order</th>
      <th>Sales per customer</th>
      <th>Delivery Status</th>
      <th>Late_delivery_risk</th>
      <th>Category Id</th>
      <th>Category Name</th>
      <th>Customer City</th>
      <th>...</th>
      <th>Order Zipcode</th>
      <th>Product Card Id</th>
      <th>Product Category Id</th>
      <th>Product Description</th>
      <th>Product Image</th>
      <th>Product Name</th>
      <th>Product Price</th>
      <th>Product Status</th>
      <th>shipping date (DateOrders)</th>
      <th>Shipping Mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>4</td>
      <td>91.250000</td>
      <td>314.640015</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>2/3/2018 22:56</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>4</td>
      <td>-249.089996</td>
      <td>311.359985</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/18/2018 12:27</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASH</td>
      <td>4</td>
      <td>4</td>
      <td>-247.779999</td>
      <td>309.720001</td>
      <td>Shipping on time</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>San Jose</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/17/2018 12:06</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>4</td>
      <td>22.860001</td>
      <td>304.809998</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/16/2018 11:45</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAYMENT</td>
      <td>2</td>
      <td>4</td>
      <td>134.210007</td>
      <td>298.250000</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.75</td>
      <td>0</td>
      <td>1/15/2018 11:24</td>
      <td>Standard Class</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>

- Check null values and deal with them
```python
# check the null values
null_value = sc.isnull().sum()
null_value[null_value>0]
```

    Customer Lname              8
    Customer Zipcode            3
    Order Zipcode          155679
    Product Description    180519
    dtype: int64

Deal with nulls:
   1. combine and create
	2. fill with zeros/ mean/median
	3. drop feature
```python
# Deal with null values
# 1.create new feature: full name
sc['Customer Full Name'] = sc['Customer Fname'] + sc['Customer Lname']
# 2.fill with 0
sc['Customer Zipcode'].value_counts()
sc['Customer Zipcode'].fillna(0, inplace=True)
# 3.drop
sc.drop(['Order Zipcode', 'Product Description'], axis=1, inplace=True)
```

- correlation analysis
```python
plt.figure(figsize=(24, 18))
sns.heatmap(sc.corr(), annot=True, cmap='coolwarm')
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208104658156.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N3ZWV0c2hhcms=,size_16,color_FFFFFF,t_70#pic_center)

Get the features of high relations and drop them.
```python
features = ['Category Id', 'Customer Zipcode', 'Order Customer Id', 'Department Id','Sales per customer', 'Order Item Id', 'Benefit per order', 'Order Item Cardprod Id','Product Card Id', 'Order Item Product Price']
sc.drop(features, axis=1, inplace=True)
```
- Information Visualization: See sales from different perspectives
1. group by market
2. group by region
3. group by product category
4. group by time
	Use time as an example:
```python
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
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208113322369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N3ZWV0c2hhcms=,size_16,color_FFFFFF,t_70#pic_center)

## FRM Model
What Is Recency, Frequency, Monetary Value (RFM)?
Recency, frequency, monetary value is a marketing analysis tool used to identify a company's or an organization's best customers by using certain measures. The RFM model is based on three quantitative factors:
1. Recency: How recently a customer has made a purchase
2. Frequency: How often a customer makes a purchase
3. Monetary Value: How much money a customer spends on purchases

RFM factors illustrate these facts:
- The more recent the purchase, the more responsive the customer is to promotions
- The more frequently the customer buys, the more engaged and satisfied they are
- Monetary value differentiates heavy spenders from low-value purchasers

Considering these facts, we will have a standard to do customer value segmentation.
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201208143553488.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N3ZWV0c2hhcms=,size_16,color_FFFFFF,t_70)

- Calculate R_value，F_value,M_value, Group by Customer Id
```python
# 1.look at the date
sc['order date (DateOrders)'] = pd.to_datetime(sc['order date (DateOrders)'])
sc['order date (DateOrders)'].max()
# Assume today is 2018-02-01

current = datetime.datetime(2018, 2, 1)
# print(current)

# 2.Define recency as the difference between the date of  last order and today: R_value
# Calculate the number orders and the total sales as Frequency and Monetary Value: M_Value, R_Value
customer_seg = sc.groupby('Customer Id').agg({'order date (DateOrders)': lambda x: (current - x.max()).days,'Order Id': lambda x: len(x), 'Sales': lambda x: x.sum()})
customer_seg.rename(columns={'order date (DateOrders)': 'R_value','Order Id': 'F_value', 'Sales': 'M_value'}, inplace=True)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R_value</th>
      <th>F_value</th>
      <th>M_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.25</th>
      <td>75.0</td>
      <td>1.0</td>
      <td>293.040008</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>159.0</td>
      <td>7.0</td>
      <td>1499.825033</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>307.0</td>
      <td>15.0</td>
      <td>2915.880065</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 3.Divide RFM into four groups in terms of value
##Function1: Deal with R_value: the smaller, the better to get R_score.
##Function2: Deal with F_value and M_value: the larger, the better to get F_score and M_score.
customer_seg['R_score'] = customer_seg['R_value'].apply(R_score, args=('R_value', quantiles))
customer_seg['F_score'] = customer_seg['F_value'].apply(FM_score, args=('F_value', quantiles))
customer_seg['M_score'] = customer_seg['M_value'].apply(FM_score, args=('M_value', quantiles))

## function3: do customer segementation interms of scores.
customer_seg['Customer Segmentation'] = customer_seg.apply(RFM_User, axis=1)
customer_seg.head(10)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R_value</th>
      <th>F_value</th>
      <th>M_value</th>
      <th>R_score</th>
      <th>F_score</th>
      <th>M_score</th>
      <th>Customer Segmentation</th>
    </tr>
    <tr>
      <th>Customer Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>792</td>
      <td>1</td>
      <td>499.950012</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Normally Detaining</td>
    </tr>
    <tr>
      <th>2</th>
      <td>136</td>
      <td>10</td>
      <td>1819.730034</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>Importantly Valuable</td>
    </tr>
    <tr>
      <th>3</th>
      <td>229</td>
      <td>18</td>
      <td>3537.680094</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>Importantly Retaining</td>
    </tr>
    <tr>
      <th>4</th>
      <td>380</td>
      <td>14</td>
      <td>1719.630030</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>Importantly Retaining</td>
    </tr>
    <tr>
      <th>5</th>
      <td>457</td>
      <td>7</td>
      <td>1274.750023</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>Normally Detaining</td>
    </tr>
    <tr>
      <th>6</th>
      <td>646</td>
      <td>15</td>
      <td>3259.510025</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>Importantly Retaining</td>
    </tr>
    <tr>
      <th>7</th>
      <td>220</td>
      <td>22</td>
      <td>5569.480106</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>Importantly Retaining</td>
    </tr>
    <tr>
      <th>8</th>
      <td>126</td>
      <td>19</td>
      <td>3763.500042</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>Importantly Valuable</td>
    </tr>
    <tr>
      <th>9</th>
      <td>140</td>
      <td>14</td>
      <td>3229.680056</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>Importantly Valuable</td>
    </tr>
    <tr>
      <th>10</th>
      <td>307</td>
      <td>8</td>
      <td>1264.790012</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>Normally Retaining</td>
    </tr>
  </tbody>
</table>
</div>

## Fraud Prediction

Risk management is very import to a company. Nowadys, access to Internet is much easier than before. Besides finance industry, Internet companies have also been experienced a lot fraud cases and lost money there. Especially for e-commerce companies, they shoulder most responsibility in bridging consumers and suppliers. Frequent and large fraud orders could be the headache. Therefore, there's always a risk management department in Internet companies. The job is to find fraud cases to help company cut losses.

As this dataset contains a *suspected_fraud* feature, I had decided to train a model for fraud detection.

- Data preprocess
1. get label
2. handle nulls
3. label encoder categorical features

```python
from sklearn.preprocessing import LabelEncoder
# Create Target label
train['fraud'] = np.where(train['Order Status']=='SUSPECTED_FRAUD',1,0)

# drop unnecessary categorical columns
drop_cols = ['Customer Email','Customer Fname', 'Customer Lname','Customer Password', 'Product Image','Product Status','Late_delivery_risk']
train.drop(drop_cols, axis=1, inplace=True)

# check nulls before label encoder
null_index = train[train['Customer Full Name'].isnull()]
train.dropna(subset=['Customer Full Name'],inplace=True)

# label encoder
label_encoder = LabelEncoder()
for col in categorical_cols:train[col] = label_encoder.fit_transform(train[col])
train[categorical_cols]
```
- data split and standardization
```python
# split train and test data
from sklearn.model_selection import train_test_split
fraud_x_train,fraud_x_test, fraud_y_train, fraud_y_test = train_test_split(fraud_x,fraud_y,test_size = 0.2)
# standardization
sscaler = StandardScaler()
fraud_x_train = sscaler.fit_transform(fraud_x_train)
fraud_x_test = sscaler.transform(fraud_x_test)
```
- Build a function for training different models
```python
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
```
- Use 7 models to fit 
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Logistic Regression
lr_fraud = LogisticRegression()
result = model_stats(lr_fraud, fraud_x_train,fraud_x_test,fraud_y_train,fraud_y_test)
accuracy.append(result[0])
recall.append(result[1])
f1.append(result[2])

# Result table
models = ['LogisticRegression','LinearSVC','GaussianNB','KNeighborsClassifier','DecisionTree','RandomForest','LDA']
re = pd.DataFrame({'Models':models,'Accuracy':accuracy,'Recall':recall,'F1 Score':f1})
```

    model is ： LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)
    accuracy of fraud：97.85059413345152%
    recall of fraud：53.96825396825397%
    F1 score of fraud：30.46594982078853%
    AUC offraud：76.10254656610977%
    confusion matrix of fraud：
    [[35157   631]
     [  145   170]]

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Models</th>
      <th>Accuracy</th>
      <th>Recall</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>0.978506</td>
      <td>0.539683</td>
      <td>0.304659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LinearSVC</td>
      <td>0.977924</td>
      <td>0.506667</td>
      <td>0.276113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GaussianNB</td>
      <td>0.980223</td>
      <td>0.528713</td>
      <td>0.691710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNeighborsClassifier</td>
      <td>0.981691</td>
      <td>0.833333</td>
      <td>0.346192</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DecisionTree</td>
      <td>0.994848</td>
      <td>0.881041</td>
      <td>0.884328</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RandomForest</td>
      <td>0.996815</td>
      <td>0.948953</td>
      <td>0.926518</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LDA</td>
      <td>0.979974</td>
      <td>0.560559</td>
      <td>0.499654</td>
    </tr>
  </tbody>
</table>
</div>

- From the table we can see it clearly, Random forest performs best for this dataset.  And for LogisticRegression may have a shrinkage problem.

## Conclusion

In real world, the job of Business Intelligence Engineer is much more complex and broad. Coding skills and analysis tools are the basic requirement. What is more important is the insight and creativity. 

1. Different dataset

There's tons of data everyday here, what can we do to make use of it is to relate real problems to the data. Also, in different inustries there are different problems to solve. So, the requirements for data are diverse too. 

For an e-commerce company, tasks are complex. For instance, this kind of Supply Chain data could be used for prediction, fraud detection and so on. It could help solve problems along the whole supply chain. However, product like YouTube and Spotify will focus on customer's tastes to do customized recommendation. What they want could be data stream because customers' tastes and interests are changeable.



2. Different tasks and goals

As we mentioned above, goals of data analytics are different. Common goals are high efficiency, high quality, low costs, low expense, more convenience and more profits. But in real operation, main focus could be different. 

Data Analyst in after sale service department will try to conclude problems in customer feedback and complaints and convey the information or give suggestions. But another data analyst in marketing department will try to track popularity and customers' tastes as well.


- Supplement
There's  more analysis using this dataset. If you are interested, you can check the code. Topics include late delivery prediction, sales prediction.

