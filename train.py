import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from utils.preprocessing import scale_data, create_sequences\

ticker = "TCS.NS"

start_date = "2021-01-01"
end_date = None

MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"
print("Downloading data")
df = yf.download(ticker , start = start_date, end = end_date)

os.makedirs("data/raw", exist_ok=True)
raw_data_path = f"data/raw/{ticker}.csv"
df.to_csv(raw_data_path)
print(f"Raw dataset saved at: {raw_data_path}")
close_data = df[['Close']]
print(close_data.tail())

train_size = int(len(close_data) * 0.8)

train_data = close_data[:train_size]
test_data = close_data[train_size:]

scaler, scaled_train, scaled_test = scale_data(
    train_data,
    test_data
)
os.makedirs("data/processed", exist_ok=True)

scaled_train_df = pd.DataFrame(
    scaled_train,
    columns=['Close']
)

scaled_test_df = pd.DataFrame(
    scaled_test,
    columns=['Close']
)

scaled_train_df.to_csv(
    "data/processed/scaled_train.csv",
    index=False
)

scaled_test_df.to_csv(
    "data/processed/scaled_test.csv",
    index=False
)

print("Processed datasets saved.")

joblib.dump(scaler, SCALER_PATH)

print("Scaler saved.")

x_train, y_train = create_sequences(
    scaled_train
)
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

model = Sequential()
model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(x_train.shape[1], 1)
    )
)
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()
early_stop = EarlyStopping(monitor='loss',patience=10,restore_best_weights=True)

print("Training model...")
model.fit(x_train,y_train,epochs=50,batch_size=32,callbacks=[early_stop])
model.save(MODEL_PATH)
print("Model saved successfully.")