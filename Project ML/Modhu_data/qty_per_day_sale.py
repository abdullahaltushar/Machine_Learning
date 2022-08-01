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


data=pd.read_csv('sale_data1.csv')
data=data[['Date','Amount','Qty','Rate']]
data=data.groupby("Date").sum()
data=data.reset_index()


train_dates=pd.to_datetime(data['Date'])

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
n_past = 14
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
n_days_for_prediction=60
#n_past = 14

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='1d').tolist()
prediction = model.predict(trainX[-n_days_for_prediction:]) 


prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,1]


# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Qty':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


original = data[['Date', 'Qty']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2022-01-01']

sns.lineplot(original['Date'], original['Qty'])
sns.lineplot(df_forecast['Date'], df_forecast['Qty'])


