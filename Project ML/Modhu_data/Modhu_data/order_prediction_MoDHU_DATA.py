# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:57:05 2022

@author: Administrator
"""

# import libaray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from numpy import array
import seaborn as sns

pd.set_option('display.max_columns', None)
orderdata = pd.read_excel("MODHU_DATA.xlsx", sheet_name='ORDER')
orderdata .drop(orderdata [orderdata.TRANSTP =='ANDROID APP' ].index, inplace=True)
#orderdata=orderdata[orderdata['TRANSTP']=='ANDROID APP']
issuedata = pd.read_excel("MODHU_DATA.xlsx", sheet_name='ISSUE')
saledata = pd.read_excel("MODHU_DATA.xlsx", sheet_name='SALE')
buydata= pd.read_excel("MODHU_DATA.xlsx", sheet_name='BUY')


data=orderdata[['TRANSDT','AMOUNT','QTY','RATE']]
#data['TRANSDT']=pd.to_datetime(data['TRANSDT'], format='%Y-%m-%d %H:%M:%S')
data=data.groupby("TRANSDT").sum()
data=data.reset_index()


train_dates=pd.to_datetime(data['TRANSDT'])

cols=list(data)[1:4]

print('Training set shape == {}'.format(train_dates.shape))
print('Featured selected: {}'.format(cols))

 # Data pre-processing
df_for_training=data[cols].astype(float)
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

trainX = []
trainY = []

n_future =1
n_past = 3
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 1])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

model = Sequential()
model.add(LSTM(units=50,activation='relu', return_sequences=True,  input_shape=(trainX.shape[1], trainX.shape[2])))

model.add(LSTM(units=50,activation='relu',return_sequences=True))

model.add(LSTM(units=50,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()

#start Training


history = model.fit(trainX, trainY, epochs=100,batch_size=64,validation_split=0.1, verbose=1)
n_days_for_prediction=28
#n_past = 14

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='1d').tolist()
prediction = model.predict(trainX[-n_days_for_prediction:]) 


prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,1]


# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'TRANSDT':np.array(forecast_dates), 'QTY':y_pred_future})
df_forecast['TRANSDT']=pd.to_datetime(df_forecast['TRANSDT'])


original = data[['TRANSDT', 'QTY']]
original['TRANSDT']=pd.to_datetime(original['TRANSDT'])
#original = original.loc[original['TRANSDT'] >= '2022-01-01']

sns.lineplot(original['TRANSDT'], original['QTY'])
sns.lineplot(df_forecast['TRANSDT'], df_forecast['QTY'])


########################predict Amount###################################
train1X = []
train1Y = []

n_future =1
n_past = 3
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    train1X.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    train1Y.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

train1X, train1Y = np.array(train1X), np.array(train1Y)

print('train1X shape == {}.'.format(train1X.shape))
print('train1Y shape == {}.'.format(train1Y.shape))

history = model.fit(train1X, train1Y, epochs=100,batch_size=64,validation_split=0.1, verbose=1)

prediction1 = model.predict(trainX[-n_days_for_prediction:]) 
prediction_copies = np.repeat(prediction1, df_for_training.shape[1], axis=-1)
y_pred_future1 = scaler.inverse_transform(prediction_copies)[:,0]


df_forecast = pd.DataFrame({'TRANSDT':np.array(forecast_dates), 'AMOUNT':y_pred_future1})
df_forecast['TRANSDT']=pd.to_datetime(df_forecast['TRANSDT'])


original = data[['TRANSDT', 'AMOUNT']]
original['TRANSDT']=pd.to_datetime(original['TRANSDT'])
#original = original.loc[original['TRANSDT'] >= '2022-01-01']

sns.lineplot(original['TRANSDT'], original['AMOUNT'])
sns.lineplot(df_forecast['TRANSDT'], df_forecast['AMOUNT'])