#Intership project at KM Cube Asset Management
#participants : Kostas Metaxas CEO of KM Cube Asset Management 
#               and Nikolaos Kalivas as an assistant from Computer Science
#               Department at Athens University of Economics and Business.
import time
import json
import csv
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
import dateparser
import pytz
import binance
from binance.client import Client
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)

def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds
    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str
    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms

def get_historical_klines(symbol, interval, start_str, end_str=None):
    """Get Historical Klines from Binance
    See dateparse docs for valid start and end string formats http://dateparser.readthedocs.io/en/latest/
    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    :param symbol: Name of symbol pair e.g BNBBTC
    :type symbol: str
    :param interval: Biannce Kline interval
    :type interval: str
    :param start_str: Start date string in UTC format
    :type start_str: str
    :param end_str: optional - end date string in UTC format
    :type end_str: str
    :return: list of OHLCV values
    """
    # create the Binance client, no need for api key
    client = Client("", "")

    # init our list
    output_data = []

    # setup the max limit
    limit = 1000

    # convert interval to useful value in seconds
    timeframe = interval_to_milliseconds(interval)

    # convert our date strings to milliseconds
    start_ts = date_to_milliseconds(start_str)

    # if an end time was passed convert it
    end_ts = None
    if end_str:
        end_ts = date_to_milliseconds(end_str)

    idx = 0
    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False
    while True:
        # fetch the klines from start_ts up to max 500 entries or the end_ts if set
        temp_data = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_ts,
            endTime=end_ts
        )

        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_data):
            symbol_existed = True

        if symbol_existed:
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            start_ts = temp_data[len(temp_data) - 1][0] + timeframe
        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(1)

    return output_data

#Call our Binance Client API, for Certain period of time
klines = get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1HOUR, "22 Feb, 2018", "22 Feb, 2019")
df = pd.DataFrame(klines)
coinTble = df[[0,1,2,3,4,5]]
coinTble.columns = ['time','price_open','price_high','price_low','price_close','volume']
coinTble["time"] = pd.to_datetime(coinTble["time"] , unit = 'ms')

def float_maker(coinTble): #make values from object to float
  coinTble["price_open"] = coinTble["price_open"].str.replace(',', '').astype(float)
  coinTble["price_high"] = coinTble["price_high"].str.replace(',', '').astype(float)
  coinTble["price_low"] = coinTble["price_low"].str.replace(',', '').astype(float)
  coinTble["price_close"] = coinTble["price_close"].str.replace(',', '').astype(float)
  coinTble["volume"] = coinTble["volume"].str.replace(',', '').astype(float)
  return()

float_maker(coinTble)

#Create a new dataframe with only the 'price_high column'
data = coinTble.filter(['price_high'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model(we use 80%)
training_data_len = len(dataset) * .8
training_data_len = int(training_data_len)
#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
#Split the data x_train and y_train data sets
x_train = []
y_train = []

for i in range (60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

# Initialising the RNN(recurrent neural network )
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, epochs = 200, batch_size = 512)

#Create the testing Data set
#Create a new array containg scaled values from the remaining 0.2 data which left to test the model
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = regressor.predict(x_test)
predictions = scaler.inverse_transform(predictions)

MSE = np.square(np.subtract(y_test,predictions)).mean() 
#print('\nMSE = ', MSE)
rmse=np.sqrt(np.mean(((predictions - y_test)**2)))
print('\nRMSE = ', rmse)

#prepare the results as an output of csv files
train = data[:training_data_len]
valid = data[training_data_len:]
Date = coinTble['time']
Date = Date[training_data_len:]
#Transform to pandas tables and then to csv files
Prediction_table = pd.DataFrame(predictions)
Prediction_table.columns = ['Predictions']
Prediction_table["Predictions"] = Prediction_table["Predictions"].astype(int)
Date_table = pd.DataFrame(Date)
Valid_table = pd.DataFrame(valid)
Valid_table["price_high"] = Valid_table["price_high"].astype(int)


#Save our Predicted_values,Valid_values and their DateTime in a seperate csv file
Prediction_table.to_csv("Predicted_output.csv", index=False, header=True)
Date_table.to_csv("Date_output.csv", index=False, header=True)
Valid_table.to_csv("Valid_output.csv", index=False, header=True)
