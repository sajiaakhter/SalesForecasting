# Future Sales Prediction 

# Application: 
In this project, we need to predict total sales for every product in every store for the next month. 

Data:  A challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. 
The dataset can be used from this Kaggle competition: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales


# Introduction

Sales forecasting is a critical process that enables companies to anticipate future sales performance and plan accordingly. It involves analyzing past sales data, market trends, and other relevant information to estimate future sales volumes and revenue. With accurate sales forecasts, companies can make informed decisions about product development, inventory management, and resource allocation.

In this notebook, I create a prediction model to predict monthly sales for every items in every retail stores for the largest Russian software firms - 1C Company. They provided a time-series dataset consisting of daily sales data (https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales).


# Dataset Description

The dataset has daily sales data for different items in different shops from January 2013 to October 2015. There are total 60 shops and 22170 items. All items are grouped into 84 categories. 

We need to predict the total number of products sold in every shop for the test set for the month of November 2015. There are 214200 pairs of items and shops in test set. However, in the training set, 102796 pairs of items and shops are missing. That means,
	- The item is in the training set, but not with the corresponding shop
	- The item is absent in the training set


# Data Loading and Preprocessing

After loading the provided data, the dataframe is cleaned with standard steps like drop the negative count for item sales, convert the date column to the datetime dtype for enabling datetime operations.


# Exploratory Data Analysis

I did EDA on sales data, category-items data and shop data. Observations and findings with graphs are in the Jupiter Notebook.



# Feature Scaling

I did feature scaling on total item sales using StandardScaler, RobustScaler, MinMaxScaler.


# Modeling
For modeling I have used ARIMA and XGBoost. 

ARIMA

First I used ARIMA model. Since I need to predict total sales for every possible combination of shops and items, I create the training sets differently. For every combination of shops and items in the test datasets:
	- First, I make a training set having total monthly sales for 33 months (from January 2013 to October 2015) using training datasets.
	- Then, I create a model using that training set. 
	- Finally, I predict the next month’s sales using this particular model for that pair of shop and item. 

The main problem in this approach is that 102796 pairs of shop and item are missing in the training dataset. So I cannot make a model to predict them.

XGBoost

To overcome the issue using ARIMA, I tried XGBoost. First, I create a training matrix that has the total month's sales for all the combination of shops and items for every month in the training data period. Then the test data are concatenated to the end of the training matrix to predict the next month’s sales.


# Performance Analysis

The RMSE using ARIMA is 2.04. Since we cannot predict almost half of the data, the RMSE is not good using this approach.

Using XGBoost, the RMSE is 1.077, which is better than ARIMA.

To improve the model performance, I tried moving average of total sales in different time range. Here’s the performance chart using different approaches.

Total Sales (original data) -> RMSE (training set) = 0.944 and RMSE (test set) = 1.077
simple moving average of total sales (min_periods = 3) -> RMSE (training set) = 0.63 and RMSE (test set) = 1.09 
simple moving average of total sales (min_periods = 12) -> RMSE (training set) = 0.54 and RMSE (test set) = 1.14
Cumulative moving average of total sales -> RMSE (training set) = 0.372 and RMSE (test set) = 1.17
Exponential moving average of total sales (min_periods = 12) -> RMSE (training set) = 0.515 and RMSE (test set) = 1.1
Exponential moving average of total sales (min_periods = 3) -> RMSE (training set) = 0.6789 and RMSE (test set) = 1.04

So, the model gives the best prediction (where RMSE is 1.04) using exponential moving average of total sales for time period 3.

