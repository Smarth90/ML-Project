from fastapi import FastAPI
from predict import predict_stock

app = FastAPI(title= "Stock Prediction API",
              description= "LSTM based stock prediction API",
            )

@app.get("/")
def home():
    return{
        "message" : "Stock prediction API running!"
    }

@app.get("/predict/{ticker}")
def predict(ticker: str):
    actual, predicted = predict_stock(ticker)
    return {
        "ticker": ticker,
        "actual": actual.flatten().tolist(),
        "predicted": predicted.flatten().tolist()
    }