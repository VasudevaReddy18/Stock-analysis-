# AI-Powered Stock, Crypto & Market Sentiment Predictor

By Vasudeva Reddy | Wipro CPPE Program Project

---

## Project Overview

This project is an **AI-powered financial sentiment analysis and market prediction system** that analyzes **Reddit posts, Google News headlines, and market data** to forecast short-term stock and cryptocurrency price movements.

The system integrates **Natural Language Processing (NLP)**, **Machine Learning (ML)**, and **financial data aggregation** from multiple APIs — creating a complete decision-support tool for investors and analysts.

---

## Objectives

* Predict **next-day market movement (Up or Down)** using real-time sentiment and price signals.
* Analyze **social sentiment** from **Reddit** and **Google News** without paid APIs.
* Enable support for both **Stock** and **Crypto** tickers (AAPL, TCS.NS, BTC-USD, etc.).
* Provide **Explainable AI** using SHAP values for model transparency.
* Create a **resilient, offline-friendly app** that works even when Yahoo Finance APIs fail.

---

## Tech Stack

* **Frontend / App Framework:** Streamlit
* **Backend / Data:** Python 3.11, Pandas, NumPy
* **APIs:** Yahoo Finance, Stooq (via `pandas-datareader`), CoinGecko
* **Social Data:** Reddit API (PRAW), Google News
* **ML / NLP:** scikit-learn, Transformers (DistilBERT), NLTK (VADER)
* **Visualization:** Plotly, Matplotlib, SHAP
* **Deployment:** Docker, Streamlit Cloud / Render
* **Secrets Management:** `.env` with python-dotenv

---

## Core Features

* Multi-source data fusion: Reddit, Google News, and market price data.
* Triple API fallback: Yahoo → Stooq → CoinGecko for uninterrupted operation.
* NLP Sentiment Analysis: DistilBERT Transformer (fallback: NLTK VADER).
* ML Prediction: Random Forest Classifier for binary Up/Down trend.
* Explainability: SHAP visualizations for feature importance.
* Backtesting: Compares AI strategy vs actual market returns.
* Crypto Support: Works for BTC, ETH, DOGE with CoinGecko fallback.
* Auto Ticker Detection: Recognizes NSE, BSE, NASDAQ tickers automatically.
* Production Stability: Fully handles missing data, malformed columns, and rate-limits.

---

## System Workflow

1. **Data Collection**

   * Reddit posts fetched via `praw`
   * Google News articles scraped via `GoogleNews`
   * Market data fetched via Yahoo Finance (primary), Stooq (secondary), CoinGecko (crypto)

2. **Sentiment Analysis**

   * Text cleaned and analyzed via Transformer pipeline
   * If Transformers unavailable → fallback to VADER
   * Average daily sentiment computed

3. **Data Merging**

   * Daily sentiment merged with stock OHLCV data
   * Price movement labels generated based on next-day close

4. **Model Training**

   * Random Forest Classifier trained on features: `avg_sentiment`, `pct_change`, `scaled_volume`

5. **Prediction + Explainability**

   * SHAP values visualize model feature impact
   * Predicts “Up” or “Down” trend with confidence %

6. **Backtesting**

   * Calculates cumulative market vs AI strategy returns

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/stock-sentiment-app.git
cd stock-sentiment-app
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # (Windows)
# or
source venv/bin/activate   # (Linux/Mac)
```

### 3. Install dependencies

```bash
pip install --no-cache-dir -r requirements.txt
```

> If `torch` fails to install, use:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Create `.env` file (for Reddit API)

Copy `.env.template` → `.env`, then fill:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

### 5. Run the app

```bash
streamlit run stock_tweet_sentiment_app.py
```

Open the app at: `http://localhost:8501`

---

## Docker Deployment

**Build image**

```bash
docker build -t stock-sentiment-app .
```

**Run container**

```bash
docker run -p 8501:8501 stock-sentiment-app
```

Access: `http://localhost:8501`

---


## Model Details

* **Algorithm:** Random Forest Classifier
* **Input Features:** avg_sentiment, pct_change, scaled_volume
* **Target Variable:** Binary (1 = Up, 0 = Down)
* **Explainability:** SHAP (feature impact plots)
* **Evaluation Metrics:** Precision, Recall, F1-Score, Confusion Matrix

---

## Sample Use Cases

* Predict next-day trend of a stock (e.g., AAPL, TSLA, INFY.NS).
* Analyze crypto sentiment for BTC-USD, ETH-USD.
* Compare Reddit and News sentiment over time.
* Backtest sentiment-driven investment strategy.
* Research or academic demo on NLP + Financial ML.

---

## Notes & Credits

* This project is for educational/demo purposes. Do not use predictions for real trading without further validation and risk controls.
* Built with ❤️ and persistent debugging by Vasudeva Reddy for the Wipro CPPE program.
