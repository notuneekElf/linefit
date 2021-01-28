import pandas as pd
import os
import quandl
import math
import time
import numpy as np
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
#imported the necessary libraries, take care with the quandl library apparently Quandl isn't a thing and 
#sk learn is used for a couple of things the mse of our model, train_test_split cross validates and our model of choice is also courtesy of scikitlearn
#use your quandl token below
auth_tok="#######"

df=quandl.get("EOD/AAPL", trim_start="1999-04-30", trim_end="2020-12-31", authtoken=auth_tok)
#data preprocessing because there are many redundant columns in the dataframe so 

df=df[['Adj_Open','Adj_Close','Adj_High','Adj_Low','Adj_Volume']]
df['HL_PCT']=(df['Adj_High']-df['Adj_Low']/df['Adj_Close'])*100
df['PCT_change']=(df['Adj_Open']-df['Adj_Close']/df['Adj_Open'])*-100.0
df=df[['Adj_Close','Adj_Volume','HL_PCT','PCT_change']]
forecasting='Adj_Close'
#nan cleaning hehe
df.fillna(-99.999)
#this is the potential range of values the model will predict, here at 10%
forecasted=int(math.ceil(0.01*len(df)))
df['label']=df[forecasting].shift(-forecasted)
df.dropna(inplace=True)
X=np.array(df.drop(['label'],1))
X=preprocessing.scale(X)
Y=np.array(df['label'])
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
clf=LinearRegression()
clf.fit(X_train, Y_train)
forecast_set=clf.predict(X)
mean_squared_error(forecast_set,forecasted)