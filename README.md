# VFV.TO Stock Market Predictor 📈

This project uses machine learning to predict whether VFV.TO (Vanguard S&P 500 ETF traded in CAD) will go up or down the next trading day. It includes a training notebook and a live Streamlit web app.

## 🔍 Features

- Uses historical VFV.TO data from Yahoo Finance via `yfinance`
- Trains a Random Forest classifier with rolling trend & ratio features
- Performs realistic backtesting using a sliding time window
- Deploys to a live interactive app with Streamlit

## 🚀 Live App

👉 [Try it here](https://vfv-stock-predictor.streamlit.app)

## 📂 Files

- `vfv_market_prediction.py`: Main Streamlit app
- `VFV_market_prediction.ipynb`: Original Colab development notebook
- `requirements.txt`: App dependencies

## 🧠 Model Details

- Classifier: `RandomForestClassifier`
- Features: Price, volume, rolling trends, and ratios
- Threshold: Custom probability threshold for predicting "up"

## 📈 Example Output

The app shows:
- Precision, accuracy, recall, and F1 score
- Predicted vs. actual direction
- A table of recent model outputs

## 🛠️ Setup (Local)

```bash
pip install -r requirements.txt
streamlit run vfv_market_prediction.py
