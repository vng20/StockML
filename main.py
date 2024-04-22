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
#plt.figure(figsize = (16,8))
#plt.title('Close Price Hist')
#plt.plot(data_frame['Close'])
#plt.xlabel('Date')
#plt.ylabel('Close Price USD')
#plt.show()

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

#Create training data set
#Use scaled data set
train_data = scaled_data[0:training_data_len, :]
#Split into x_train and y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#LSTM model needs 3D shape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create testing data set
test_data = scaled_data[training_data_len - 60: , :]
#Create datasets x_text and y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get models predicted price values
#Get the root mean sqaured error
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)

#plt data
train = data_frame_train[:training_data_len]
valid = data_frame_train[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize= (16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'])
plt.show()

#actual closing price vs predicted prices
print(valid)
margin_of_error = np.array(valid)
margin_of_error[:, 1] /= margin_of_error[:, 0]
print(margin_of_error[:, 1])
total = 0
for i in range(len(margin_of_error[:, 1])):
    total += margin_of_error[i]
avg_me = total/(len(margin_of_error[:, 1]))
print(avg_me)