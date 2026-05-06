import joblib
import yfinance as yf
import numpy as np

from tensorflow.keras.models import load_model

from utils.preprocessing import create_sequences
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"

SEQUENCE_LENGTH = 100
print("Loading model...")
model = load_model(MODEL_PATH)
print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

def predict_stock(ticker="TCS.NS"):
    print(f"Downloading latest data for {ticker}")
    df = yf.download(
        ticker,
        period="2y"
    )
    close_data = df[['Close']]
    scaled_data = scaler.transform(close_data)
    x_test, y_test = create_sequences(
        scaled_data
    )
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(
        predictions
    )
    actual_prices = scaler.inverse_transform(
        y_test
    )
    return actual_prices, predictions
if __name__ == "__main__":
    actual, predicted = predict_stock()
    print()
    print("ACTUAL:")
    print(actual[:5])
    print()
    print("PREDICTED:")
    print(predicted[:5])