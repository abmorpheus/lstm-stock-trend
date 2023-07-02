import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = "2010-01-01"
end = "2021-12-31"

st.title('Stock Trend Prediction')

user_input = "GOOGL"
user_input = st.text_input('Enter Stock Ticker')
df = yf.download(user_input, start, end)

# Describe data
st.subheader('Data from 2010-2021')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

# Moving Average
st.subheader('Closing Price vs Time Chart with 100 days Moving Average')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)

train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

# Load model
model = load_model('lstm_model.h5')

# Testing

past_100_days = train.tail(100)

final_df = past_100_days.append(test, ignore_index = True)
final_df.head()

input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  X_test.append(input_data[i-100:i])
  y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

y_preds = model.predict(X_test)

scale = scaler.scale_

scale_factor = 1/scale[0]
y_preds = y_preds*scale_factor
y_test = y_test*scale_factor

# Final Graph
st.subheader('Predictions vs Actual')
fig1 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'green', label = 'Original Price')
plt.plot(y_preds, 'red', label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)