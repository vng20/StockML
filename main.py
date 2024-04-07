# This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM) to predict the closing stock price of a corporation using
# the past 60 day stock price.

import math
import pandas_datareader as web
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import yfinance as yf

# Get the stock quote
yf.pdr_override()
data_frame = pdr.get_data_yahoo('AAPL', start='2012-01-01', end='2020-01-01')
print(data_frame)

# Get number of rows and cols
print(data_frame.shape)

#plot closing price history
plt.figure(figsize = (16,8))
plt.title('Close Price Hist')
plt.plot(data_frame['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.show()

#Create dataframe with only the Close price column
data_frame_train = data_frame.filter(['Close'])
#convert dataframe into numpy array
dataset = data_frame_train.values
#get number of rows to train model on
training_data_len = math.ceil(len(dataset) * 0.8)
print(training_data_len)

#Scale the data - Good practice to preproccess transformation scaling or normalization before neural network processing
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)
