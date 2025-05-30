# 📈 VFV.TO Stock Movement Predictor

This machine learning project predicts whether **VFV.TO** (Vanguard S&P 500 ETF, CAD) will go up or down the next trading day. It combines financial data analysis, model building, and a production-grade **Streamlit web app** to demonstrate real-world ML deployment.

---

## 🧠 Project Goals

- Create an ML model that simulates how well we could predict VFV’s next-day price direction.
- Evaluate it using **realistic backtesting**, not just train/test splits.
- Provide live predictions with confidence scores via a shareable web interface.

---

## 🛠️ Project Steps

1. **Find a dataset**  
   → Download historical data from Yahoo Finance using `yfinance`.

2. **Build a data pipeline**  
   → Add engineered features (e.g., rolling price ratios, trend counts).

3. **Train a model**  
   → Use `RandomForestClassifier` to predict next-day direction.

4. **Evaluate with backtesting**  
   → Simulate real-world usage using a sliding-window method.

5. **Deploy to the web**  
   → Build a live Streamlit app with daily predictions and metrics.

6. **Think of next steps**  
   → Improve accuracy with better features, models, or thresholds.

---

## 🔍 Features

- Historical VFV.TO data from Yahoo Finance
- Rolling feature engineering (trends, ratios)
- Realistic backtesting engine to reduce lookahead bias
- Custom thresholding for high-confidence predictions
- Live “Prediction for Tomorrow” card in the deployed app

---

## 🚀 Try the Live App

👉 [Open Streamlit App](https://vfv-stock-predictor.streamlit.app)

See model performance metrics, past predictions, and what it thinks will happen **tomorrow**.

---

## 📂 Files

- `vfv_market_prediction.py` – Main Streamlit app (production-ready)
- `VFV_market_prediction.ipynb` – Original Colab prototype
- `requirements.txt` – Dependencies for local setup

---

## 🤖 Model Details

| Component        | Details                              |
|------------------|---------------------------------------|
| Classifier       | `RandomForestClassifier`              |
| Features         | `Close`, `Volume`, rolling ratios & trends |
| Backtesting      | Walk-forward validation (300+ day steps) |
| Threshold        | Custom (e.g., 0.6+ confidence = “up”) |

---

## 📈 Output

- Model metrics: **precision**, **accuracy**, **recall**, **F1**
- Comparison plot of actual vs. predicted directions
- Recent raw predictions
- Tomorrow’s prediction with confidence score

---

## 🧪 Run Locally

```bash
pip install -r requirements.txt
streamlit run vfv_market_prediction.py
```
💡 Next Steps (Ideas for Improvement)
- Estimate hypothetical profit/loss from following predictions
- Try XGBoost or LSTM instead of Random Forest
- Add sentiment features from news headlines
- Improve feature timing (e.g., pre-market indicators)


📬 Contact

Built by Rosa Sakhapour · Feedback welcome!
