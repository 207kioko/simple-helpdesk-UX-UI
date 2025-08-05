import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas_datareader as data
from datetime import datetime
from keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from mplfinance.original_flavor import candlestick_ohlc
import random
import matplotlib.dates as mdates
from pytz import timezone
from datetime import timedelta

# Define timezone
kenya_timezone = timezone('Africa/Nairobi')
current_time_kenya = datetime.now(kenya_timezone)

# Data Description
st.title('Google Stock Prediction Model System')
user_input = st.text_input('Enter Stock Ticker', 'GOOG')
api_token = 'sk_a523cbccbdcf4a8287aa2197c860c131'
start = datetime(2022, 1, 1)
end = datetime.now()
stock_symbol = 'GOOG'
st.subheader('Data from 2022')
# Visualization
st.subheader('Closing Price vs Time chart')
plt.plot(df.close)
st.pyplot(fig)

# Moving Average (MA)
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.close)
st.pyplot(fig)

# Model Training and Prediction
data = yf.download(user_input, start='2022-01-01', end='2022-12-31')
target_column = 'Close'
data = data[[target_column]].copy()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
sequence_length = 10
sequences = []
targets = []

for i in range(len(data_scaled) - sequence_length):
    seq = data_scaled[i:i + sequence_length]
    label = data_scaled[i + sequence_length:i + sequence_length + 1]
    sequences.append(seq)
    targets.append(label)

X = np.array(sequences)
y = np.array(targets)

# Data split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Model Train
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Prediction for Today
future_date = current_time_kenya + timedelta(days=1)
future_data_today = pd.DataFrame({'Date': [future_date]})
future_data_today['Close'] = np.nan
input_data_today = data.tail(sequence_length).values.reshape(1, sequence_length, 1)
predicted_price_today = model.predict(input_data_today)
future_data_today.loc[0, 'Close'] = predicted_price_today[0, 0]

st.subheader('Predicted Price for Today:')
st.write(future_data_today[['Date', 'Close']])

# Prediction for Next Week
future_dates = pd.date_range(current_time_kenya, current_time_kenya + timedelta(days=7), freq='D')
future_data = pd.DataFrame({'Date': future_dates})
last_week_data = data.tail(sequence_length).values
future_data['Close'] = np.nan

for i in range(len(future_data)):
    input_data = last_week_data.reshape(1, sequence_length, 1)
    predicted_price = model.predict(input_data)
    future_data.loc[i, 'Close'] = predicted_price[0, 0]
    last_week_data = np.append(last_week_data[1:], predicted_price, axis=0)

st.subheader('Predicted Prices for the Next Week:')
st.write(future_data[['Date', 'Close']])

# Final Visualization
st.subheader('Prediction Graph')
def visualize_candlestick_chart_with_profit(selected_stock):
    data_stock = yf.download(selected_stock, start='2023-01-01', end='2023-12-31')
    data_stock.reset_index(inplace=True)
    data_stock['Date'] = data_stock['Date'].map(mdates.date2num)

    # Candlestick chart plot
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    candlestick_ohlc(ax, data_stock[['Date', 'Open', 'High', 'Low', 'Close']].values, width=0.6, colorup='g', colordown='r')
    plt.title(f'Candlestick Chart for {selected_stock}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    def on_scroll(event):
        if event.button == 'up':
            scale_factor = 1 / 1.5
        else:
            scale_factor = 1.5
        xlim = ax.get_xlim()
        new_xlim = [xlim[0] * scale_factor, xlim[1] * scale_factor]
        ax.set_xlim(new_xlim)
        plt.draw()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    st.pyplot(fig)  # Show the chart in Streamlit app

selected_stock = 'GOOG'
visualize_candlestick_chart_with_profit(selected_stock)