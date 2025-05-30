import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="VFV Stock Predictor", layout="centered")
st.title("📈 VFV.TO Stock Movement Predictor")
st.write("This app uses a machine learning model to predict whether VFV.TO (S&P 500 ETF) will go up or down tomorrow.")

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    vfv = yf.Ticker("VFV.TO").history(period="max")
    vfv.index = pd.to_datetime(vfv.index, utc=True)
    vfv.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
    vfv["Tomorrow"] = vfv["Close"].shift(-1)
    vfv["Target"] = (vfv["Tomorrow"] > vfv["Close"]).astype(int)
    return vfv

vfv = load_data()

# --- Feature engineering ---
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = vfv.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    vfv[ratio_column] = vfv["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    vfv[trend_column] = vfv.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

vfv = vfv.dropna(subset=vfv.columns[vfv.columns != "Tomorrow"])

# --- Model and backtesting ---
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.5).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return pd.concat([test["Target"], preds], axis=1)

def backtest(data, model, predictors, start=300, step=50):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
predictions = backtest(vfv, model, new_predictors)

# --- Display results ---
st.subheader("Model Evaluation Metrics")
st.write("These scores reflect the model's performance during historical backtesting.")

col1, col2 = st.columns(2)
col1.metric("Precision", f"{precision_score(predictions['Target'], predictions['Predictions']):.2f}")
col1.metric("Accuracy", f"{accuracy_score(predictions['Target'], predictions['Predictions']):.2f}")
col2.metric("Recall", f"{recall_score(predictions['Target'], predictions['Predictions']):.2f}")
col2.metric("F1 Score", f"{f1_score(predictions['Target'], predictions['Predictions']):.2f}")

st.subheader("Target vs Predicted Directions")
st.write("Visual comparison of actual target and model predictions (0 = down, 1 = up).")

fig, ax = plt.subplots()
predictions["Target"].plot(label="Actual", ax=ax)
predictions["Predictions"].plot(label="Predicted", ax=ax)
plt.legend()
st.pyplot(fig)

st.subheader("Raw Prediction Data")
st.dataframe(predictions.tail(20))
