## Table of contents

 - [Project information](#project-information)
 - [Data cleaning](#data-cleaning)
 - [SQL](#sql)
 - [Data visualization](#data-visualization)
 - [Dashboard](#dashboard)
 - [Recommendations](#recommendations)

# Project information
The analysis is based on supermarket sales data from a three-month period (January–March) in 2019. The supermarket has branches in three different cities. The goal of this analysis is to identify key sales trends, customer preferences, and evaluate differences in the popularity of product categories depending on location.

#### Questions this analysis aims to answer:
- What is the distribution of customers by gender and city
- What factors influence customer satisfaction levels
- Which product lines are the most and least popular in different cities
- Is there a relationship between the price of a product and the quantity in which it is purchased
- Which cities and product categories generate the highest profit
- During which hours do stores experience the highest customer traffic

```python
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import sqlite3

```

# Data cleaning


```python
df= pd.read_csv(r"D:\power_bi_projects\data\supermarket_sales.csv")
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Invoice ID</th>
      <th>Branch</th>
      <th>City</th>
      <th>Customer type</th>
      <th>Gender</th>
      <th>Product line</th>
      <th>Unit price</th>
      <th>Quantity</th>
      <th>Tax 5%</th>
      <th>Total</th>
      <th>Date</th>
      <th>Time</th>
      <th>Payment</th>
      <th>cogs</th>
      <th>gross margin percentage</th>
      <th>gross income</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>750-67-8428</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Female</td>
      <td>Health and beauty</td>
      <td>74.69</td>
      <td>7</td>
      <td>26.1415</td>
      <td>548.9715</td>
      <td>1/5/2019</td>
      <td>13:08</td>
      <td>Ewallet</td>
      <td>522.83</td>
      <td>4.761905</td>
      <td>26.1415</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>226-31-3081</td>
      <td>C</td>
      <td>Naypyitaw</td>
      <td>Normal</td>
      <td>Female</td>
      <td>Electronic accessories</td>
      <td>15.28</td>
      <td>5</td>
      <td>3.8200</td>
      <td>80.2200</td>
      <td>3/8/2019</td>
      <td>10:29</td>
      <td>Cash</td>
      <td>76.40</td>
      <td>4.761905</td>
      <td>3.8200</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>631-41-3108</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Home and lifestyle</td>
      <td>46.33</td>
      <td>7</td>
      <td>16.2155</td>
      <td>340.5255</td>
      <td>3/3/2019</td>
      <td>13:23</td>
      <td>Credit card</td>
      <td>324.31</td>
      <td>4.761905</td>
      <td>16.2155</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>123-19-1176</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Male</td>
      <td>Health and beauty</td>
      <td>58.22</td>
      <td>8</td>
      <td>23.2880</td>
      <td>489.0480</td>
      <td>1/27/2019</td>
      <td>20:33</td>
      <td>Ewallet</td>
      <td>465.76</td>
      <td>4.761905</td>
      <td>23.2880</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>373-73-7910</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Sports and travel</td>
      <td>86.31</td>
      <td>7</td>
      <td>30.2085</td>
      <td>634.3785</td>
      <td>2/8/2019</td>
      <td>10:37</td>
      <td>Ewallet</td>
      <td>604.17</td>
      <td>4.761905</td>
      <td>30.2085</td>
      <td>5.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop_duplicates() 
df= df.drop(columns=["Invoice ID","gross margin percentage"])
df.isnull().sum() 
```




    Branch           0
    City             0
    Customer type    0
    Gender           0
    Product line     0
    Unit price       0
    Quantity         0
    Tax 5%           0
    Total            0
    Date             0
    Time             0
    Payment          0
    cogs             0
    gross income     0
    Rating           0
    dtype: int64




```python
df["Time"]=pd.to_datetime(df["Time"],format="%H:%M") # changing data type
df["Date"]=pd.to_datetime(df["Date"],format="%m/%d/%Y")
```


```python
df["Full_date"] = df["Date"] + df["Time"].dt.time.astype(str).apply(pd.Timedelta) # adding new column
```


```python
df["Time"] = df["Full_date"].dt.time
df["Date"] = df["Full_date"].dt.date
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 16 columns):
     #   Column         Non-Null Count  Dtype         
    ---  ------         --------------  -----         
     0   Branch         1000 non-null   object        
     1   City           1000 non-null   object        
     2   Customer type  1000 non-null   object        
     3   Gender         1000 non-null   object        
     4   Product line   1000 non-null   object        
     5   Unit price     1000 non-null   float64       
     6   Quantity       1000 non-null   int64         
     7   Tax 5%         1000 non-null   float64       
     8   Total          1000 non-null   float64       
     9   Date           1000 non-null   object        
     10  Time           1000 non-null   object        
     11  Payment        1000 non-null   object        
     12  cogs           1000 non-null   float64       
     13  gross income   1000 non-null   float64       
     14  Rating         1000 non-null   float64       
     15  Full_date      1000 non-null   datetime64[ns]
    dtypes: datetime64[ns](1), float64(6), int64(1), object(8)
    memory usage: 125.1+ KB
    


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unit price</th>
      <th>Quantity</th>
      <th>Tax 5%</th>
      <th>Total</th>
      <th>cogs</th>
      <th>gross income</th>
      <th>Rating</th>
      <th>Full_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>55.672130</td>
      <td>5.510000</td>
      <td>15.379369</td>
      <td>322.966749</td>
      <td>307.58738</td>
      <td>15.379369</td>
      <td>6.97270</td>
      <td>2019-02-14 15:30:27.480000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.080000</td>
      <td>1.000000</td>
      <td>0.508500</td>
      <td>10.678500</td>
      <td>10.17000</td>
      <td>0.508500</td>
      <td>4.00000</td>
      <td>2019-01-01 10:39:00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.875000</td>
      <td>3.000000</td>
      <td>5.924875</td>
      <td>124.422375</td>
      <td>118.49750</td>
      <td>5.924875</td>
      <td>5.50000</td>
      <td>2019-01-24 17:58:45</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>55.230000</td>
      <td>5.000000</td>
      <td>12.088000</td>
      <td>253.848000</td>
      <td>241.76000</td>
      <td>12.088000</td>
      <td>7.00000</td>
      <td>2019-02-13 17:37:00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77.935000</td>
      <td>8.000000</td>
      <td>22.445250</td>
      <td>471.350250</td>
      <td>448.90500</td>
      <td>22.445250</td>
      <td>8.50000</td>
      <td>2019-03-08 15:29:30</td>
    </tr>
    <tr>
      <th>max</th>
      <td>99.960000</td>
      <td>10.000000</td>
      <td>49.650000</td>
      <td>1042.650000</td>
      <td>993.00000</td>
      <td>49.650000</td>
      <td>10.00000</td>
      <td>2019-03-30 20:37:00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26.494628</td>
      <td>2.923431</td>
      <td>11.708825</td>
      <td>245.885335</td>
      <td>234.17651</td>
      <td>11.708825</td>
      <td>1.71858</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


## SQL
```python
#!pip install ipython-sql
cnn = sqlite3.connect('jupyter_sql_supermarket.db')

execute = cnn.cursor()
df.to_sql('store',con=cnn, if_exists='replace')
```




    1000




```python
res = pd.read_sql("""select * from store""",con=cnn)
res.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Branch</th>
      <th>City</th>
      <th>Customer type</th>
      <th>Gender</th>
      <th>Product line</th>
      <th>Unit price</th>
      <th>Quantity</th>
      <th>Tax 5%</th>
      <th>Total</th>
      <th>Date</th>
      <th>Time</th>
      <th>Payment</th>
      <th>cogs</th>
      <th>gross income</th>
      <th>Rating</th>
      <th>Full_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Female</td>
      <td>Health and beauty</td>
      <td>74.69</td>
      <td>7</td>
      <td>26.1415</td>
      <td>548.9715</td>
      <td>2019-01-05</td>
      <td>13:08:00.000000</td>
      <td>Ewallet</td>
      <td>522.83</td>
      <td>26.1415</td>
      <td>9.1</td>
      <td>2019-01-05 13:08:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>C</td>
      <td>Naypyitaw</td>
      <td>Normal</td>
      <td>Female</td>
      <td>Electronic accessories</td>
      <td>15.28</td>
      <td>5</td>
      <td>3.8200</td>
      <td>80.2200</td>
      <td>2019-03-08</td>
      <td>10:29:00.000000</td>
      <td>Cash</td>
      <td>76.40</td>
      <td>3.8200</td>
      <td>9.6</td>
      <td>2019-03-08 10:29:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Home and lifestyle</td>
      <td>46.33</td>
      <td>7</td>
      <td>16.2155</td>
      <td>340.5255</td>
      <td>2019-03-03</td>
      <td>13:23:00.000000</td>
      <td>Credit card</td>
      <td>324.31</td>
      <td>16.2155</td>
      <td>7.4</td>
      <td>2019-03-03 13:23:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Male</td>
      <td>Health and beauty</td>
      <td>58.22</td>
      <td>8</td>
      <td>23.2880</td>
      <td>489.0480</td>
      <td>2019-01-27</td>
      <td>20:33:00.000000</td>
      <td>Ewallet</td>
      <td>465.76</td>
      <td>23.2880</td>
      <td>8.4</td>
      <td>2019-01-27 20:33:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Sports and travel</td>
      <td>86.31</td>
      <td>7</td>
      <td>30.2085</td>
      <td>634.3785</td>
      <td>2019-02-08</td>
      <td>10:37:00.000000</td>
      <td>Ewallet</td>
      <td>604.17</td>
      <td>30.2085</td>
      <td>5.3</td>
      <td>2019-02-08 10:37:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# suma z
res = pd.read_sql("""SELECT strftime('%m', Date) AS Month, ROUND(SUM(Total),2) AS "Monthly Sales", City
FROM store
GROUP BY Month, City
ORDER BY Month """, con=cnn)
res
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Monthly Sales</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01</td>
      <td>37176.0585</td>
      <td>Mandalay</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01</td>
      <td>40434.6810</td>
      <td>Naypyitaw</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01</td>
      <td>38681.1285</td>
      <td>Yangon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>02</td>
      <td>34424.2710</td>
      <td>Mandalay</td>
    </tr>
    <tr>
      <th>4</th>
      <td>02</td>
      <td>32934.9825</td>
      <td>Naypyitaw</td>
    </tr>
    <tr>
      <th>5</th>
      <td>02</td>
      <td>29860.1205</td>
      <td>Yangon</td>
    </tr>
    <tr>
      <th>6</th>
      <td>03</td>
      <td>34597.3425</td>
      <td>Mandalay</td>
    </tr>
    <tr>
      <th>7</th>
      <td>03</td>
      <td>37199.0430</td>
      <td>Naypyitaw</td>
    </tr>
    <tr>
      <th>8</th>
      <td>03</td>
      <td>37659.1215</td>
      <td>Yangon</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 
res = pd.read_sql("""SELECT Gender, City, ROUND(AVG("Total"), 2)AS Average_unit_price, ROUND(SUM(Total),2) AS Total_sales
FROM store
GROUP BY Gender,City
ORDER BY Total_sales desc""", con=cnn)
res
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>City</th>
      <th>Average_unit_price</th>
      <th>Total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>Naypyitaw</td>
      <td>346.55</td>
      <td>61685.46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>Mandalay</td>
      <td>313.35</td>
      <td>53269.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>Yangon</td>
      <td>330.86</td>
      <td>53269.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>Yangon</td>
      <td>295.71</td>
      <td>52931.20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>Mandalay</td>
      <td>326.72</td>
      <td>52928.29</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Male</td>
      <td>Naypyitaw</td>
      <td>325.89</td>
      <td>48883.24</td>
    </tr>
  </tbody>
</table>
</div>




```python

# 
res = pd.read_sql("""SELECT Payment, ROUND(AVG(Total),2) as Avg_Total, ROUND(SUM(Total),2) as Sum_Total
FROM store
GROUP BY Payment
ORDER BY Avg_Total DESC """, con=cnn)
res
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Payment</th>
      <th>Avg_Total</th>
      <th>Sum_Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cash</td>
      <td>326.18</td>
      <td>112206.57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Credit card</td>
      <td>324.01</td>
      <td>100767.07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ewallet</td>
      <td>318.82</td>
      <td>109993.11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 
res = pd.read_sql("""SELECT 
  strftime('%H', Time) AS Hour, 
  ROUND(SUM(Total),2) AS Total_sales, 
  ROUND(AVG(Total),2) AS Avg_total_sales, 
  COUNT(*) AS Number_of_transactions
FROM store
GROUP BY Hour
ORDER BY Hour;
""", con=cnn)
res
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hour</th>
      <th>Total_sales</th>
      <th>Avg_total_sales</th>
      <th>Number_of_transactions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>31421.48</td>
      <td>311.10</td>
      <td>101</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11</td>
      <td>30377.33</td>
      <td>337.53</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>26065.88</td>
      <td>292.88</td>
      <td>89</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>34723.23</td>
      <td>337.12</td>
      <td>103</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>30828.40</td>
      <td>371.43</td>
      <td>83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15</td>
      <td>31179.51</td>
      <td>305.68</td>
      <td>102</td>
    </tr>
    <tr>
      <th>6</th>
      <td>16</td>
      <td>25226.32</td>
      <td>327.61</td>
      <td>77</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17</td>
      <td>24445.22</td>
      <td>330.34</td>
      <td>74</td>
    </tr>
    <tr>
      <th>8</th>
      <td>18</td>
      <td>26030.34</td>
      <td>279.90</td>
      <td>93</td>
    </tr>
    <tr>
      <th>9</th>
      <td>19</td>
      <td>39699.51</td>
      <td>351.32</td>
      <td>113</td>
    </tr>
    <tr>
      <th>10</th>
      <td>20</td>
      <td>22969.53</td>
      <td>306.26</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# z jakich branÅ¼ sÄ… najczÄ™Å›ciej kupowane produkty
res = pd.read_sql("""SELECT "Product line", ROUND(SUM(Quantity),2) as Sum_Quantity
FROM store
GROUP BY "Product line" 
ORDER BY Sum_Quantity DESC""", con=cnn)
res
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product line</th>
      <th>Sum_Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Electronic accessories</td>
      <td>971.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Food and beverages</td>
      <td>952.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sports and travel</td>
      <td>920.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Home and lifestyle</td>
      <td>911.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fashion accessories</td>
      <td>902.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Health and beauty</td>
      <td>854.0</td>
    </tr>
  </tbody>
</table>
</div>



## Data visualization

```python
cor=df.corr(numeric_only=True)
sns.heatmap(cor,cmap="coolwarm",annot=True, linewidth=.5)
plt.show()
```


    
![png](output_13_0.png)
    


- customer satisfaction with purchases (Rating) is independent of the other analyzed factors (customers base their ratings on different aspects)
- the quantity of purchased products (Quantity) is not related to their unit price (Unit price)
- the strong correlation between tax (Tax 5%), cost of goods sold (COGS), revenue (Total), and gross income (Gross income) aligns with expected financial relationships


```python
plt.figure(figsize=(9, 5))

sns.barplot(data=df, x="Product line", y="Total",edgecolor="black",errorbar=None,estimator=np.size,hue="City",palette="Accent")
plt.xticks(rotation=30,size=9)
r=range(0,25000,5000)
#plt.yticks(r, [f'{x/1000:.0f}k' for x in r])
plt.title("Total customers by product line and city")
plt.ylabel("Customers")
plt.xlabel("Product line")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()

grouped = df.groupby("Product line")["Total"].count().reset_index()
grouped = grouped.sort_values(by="Total",ascending=False)
grouped
```


    
![png](output_15_0.png)
    





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product line</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Fashion accessories</td>
      <td>178</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Food and beverages</td>
      <td>174</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Electronic accessories</td>
      <td>170</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sports and travel</td>
      <td>166</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Home and lifestyle</td>
      <td>160</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Health and beauty</td>
      <td>152</td>
    </tr>
  </tbody>
</table>
</div>



- the chart presents the variation in product category popularity across different cities, indicating distinct shopping trends in each location
- the most popular categories are Fashion accessories (178 purchased products), Food and beverages (174), and Fashion accessories (170)
- the least popular category is Health and beauty, with a total of 152 products sold
- Naypyitaw stands out for its high demand for Food and beverages and Fashion accessories, which recorded the highest sales in this city. Meanwhile, Home and lifestyle and Sports and travel are significantly less popular, with around 20 fewer purchases than in other cities
- in Yangon, Home and lifestyle products are the most frequently purchased, achieving the highest sales among all cities. At the same time, Health and beauty is the least popular category in this location
- in Mandalay, Health and beauty dominates, with sales exceeding those in other cities


```python
plt.figure(figsize=(12,5))
df["Price bins"] = pd.cut(df["Unit price"], bins=30)
grouped = df.groupby("Price bins",observed=False)["Quantity"].sum().reset_index()
sns.barplot(data=grouped,x="Price bins",y="Quantity",estimator=np.sum,hue="Quantity",palette="flare",edgecolor="black")
plt.xticks(rotation=30,size=9, ha='right')
plt.title("Total purchased product quantity by price bins")
plt.show()
```


    
![png](output_19_0.png)
    


- there is no strong correlation between the unit price of a product and the quantity purchased by a customer
- the highest number of products sold (320 units) falls within the highest price range [96.9-99.9 USD], making this range stand out significantly compared to others. Customers also frequently chose products in the 19-22 USD and 73-76 USD price ranges


```python

```

# Dashboard
![png](dashboard_1.png)

## Key information
The dashboard presents revenue and cost of sales for each day of the month, with filtering options for month and city. It includes a summary table by product line and a chart displaying aggregated revenue on an hourly scale.
- total revenue: $322.97K
- total cost of sales: $307.59K
- total gross profit: $15.38K
  
## Trend analysis
- on days such as the 4th, 13th, 18th, and 22nd of each month, there are declines in revenue. On the other hand, revenue increases are observed on 7th-9th and 15th. Declines and increases in revenue alternate throughout the month
- the most profitable city is Naypyitaw, where the gross profit amounted to $5.27K. Naypyitaw's highest earnings come from the Food and Beverages category (about $1.1K) and Fashion Accessories (around $1K)
- in other cities, the gross profit is $5.06K
- low revenues are observed in the afternoon hours (3:30 PM - 6:30 PM), while the highest earnings occur in the evening (7:00 PM - 7:30 PM), suggesting that customers tend to do more shopping after work.
- the highest gross profit was recorded in January ($5.54K), while the lowest was in February ($4.63K)
  
![png](dashboard_2.png)

- the highest revenues were recorded on Saturdays ($56K in total), followed by Tuesdays ($51K)
- mondays have the lowest revenues ($38K), which could be due to fewer customers after weekend shopping
- sales on weekdays (Wednesday, Thursday, Friday) and Sundays remain at a stable level of around $44K

## Sales analysis by product line
- the differences between revenues across various categories are minor, with the highest earnings coming from Food and Beverages ($56.1K)
- the lowest gross profit is recorded in Health and Beauty ($2.3K) due to low sales, while other categories generate around $2.5K-$2.6K, suggesting stable sales across those lines
  
## Recommendations
- #### increase the number of customers during times or days with lower traffic
  - introduce promotions during times when sales are lower to boost customer activity
  - implement a loyalty program where customers earn points for every purchase, which can later be exchanged for rewards
- #### encourage more frequent purchases from less profitable categories (especially Health and Beauty)
  - expand the product offering by introducing new, popular products to attract more customers
  - run targeted marketing campaigns on social media platforms like TikTok and Instagram. This could include sending products to influencers with high visibility
  - offer personalized promotions (sent via SMS) for customers who have an account in the store, making the offers more attractive
  - regularly update the store layout and display posters to capture the attention of customers
- #### invest in profitable product lines in cities with the highest performance to maintain and increase customer interest
  - expand the range of products — for example, if customers are interested in Fashion Accessories in Naypyitaw, consider introducing more expensive and luxurious products in that category
  - increase warehouse capacity to avoid stockouts of popular items and create space for new products (as mentioned in the previous point)
  - organize local events and collaborate with influencers. This could include hosting themed events, workshops at supermarkets, or inviting influencers to be brand ambassadors to further engage customers and raise brand awareness
