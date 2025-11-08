import os
import datetime
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf
import shap
import praw
from dotenv import load_dotenv
from GoogleNews import GoogleNews
import pandas_datareader.data as web
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load environment variables
load_dotenv()

# Reddit credentials (.env required)
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Sentiment setup
USE_TRANSFORMERS = False
try:
    from transformers import pipeline
    sentiment_pipe = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    USE_TRANSFORMERS = True
except Exception:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    vader = SentimentIntensityAnalyzer()

# Streamlit UI setup
st.set_page_config(layout="wide", page_title="AI Stock & Crypto Sentiment Predictor")
st.title("📈 AI-Powered Stock, Crypto & Market Sentiment Predictor")
st.caption("Automatically analyzes sentiment + market trends and predicts short-term movement.")

# Sidebar
st.sidebar.header("Configuration")
stock_ticker = st.sidebar.text_input("Symbol (AAPL / TCS / BTC-USD)", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=datetime.date.today() - datetime.timedelta(days=90))
end_date = st.sidebar.date_input("End date", value=datetime.date.today())
max_posts = st.sidebar.number_input("Max posts per source", 50, 1000, 200, 50)
test_ratio = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random seed", value=42)

# ---------------------------
# Helper functions
# ---------------------------
def auto_correct_ticker(ticker):
    t = ticker.strip().upper()
    indian = {"TCS", "INFY", "RELIANCE", "HDFCBANK", "SBIN", "ITC", "ICICIBANK", "WIPRO", "AXISBANK", "LT", "ONGC"}
    crypto = {"BTC-USD", "ETH-USD", "DOGE-USD"}
    if t in crypto: return t
    if t in indian and not t.endswith(".NS"): return t + ".NS"
    return t

def safe_yf_download(ticker, start_date, end_date):
    candidates = [ticker]
    if not ticker.endswith((".NS", ".BO", "-USD")):
        candidates += [ticker + ".NS", ticker + ".BO"]

    # --- Yahoo Finance ---
    for cand in candidates:
        try:
            st.info(f"📡 Downloading {cand} from Yahoo Finance...")
            df = yf.download(cand, start=start_date, end=end_date,
                             progress=False, threads=False, auto_adjust=True)
            if df is not None and not df.empty:
                st.success(f"✅ Yahoo Finance data fetched for {cand}")
                return df, cand
            else:
                st.warning(f"⚠️ Yahoo returned empty data for {cand}")
        except Exception as e:
            st.warning(f"⚠️ Yahoo error for {cand}: {e}")

    # --- Stooq backup ---
    try:
        st.info("🔁 Trying Stooq backup source...")
        df = web.DataReader(ticker, "stooq", start=start_date, end=end_date)
        if df is not None and not df.empty:
            st.success("✅ Backup source (Stooq) success.")
            df = df.sort_index()
            return df, ticker
        else:
            st.warning("⚠️ Backup source returned empty.")
    except Exception as e:
        st.warning(f"⚠️ Backup source failed: {e}")

    # --- CoinGecko fallback for crypto ---
    try:
        if "-USD" in ticker:
            st.info(f"💰 Fetching crypto data from CoinGecko for {ticker}...")
            crypto_ids = {"BTC-USD": "bitcoin", "ETH-USD": "ethereum", "DOGE-USD": "dogecoin"}
            coin_id = crypto_ids.get(ticker.upper(), "bitcoin")
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {"vs_currency": "usd", "days": "90", "interval": "daily"}
            r = requests.get(url, params=params)
            if r.status_code == 200:
                data = r.json()
                if "prices" in data:
                    df = pd.DataFrame(data["prices"], columns=["timestamp", "Close"])
                    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
                    df["Volume"] = 0
                    df = df[["Date", "Close", "Volume"]]
                    st.success(f"✅ CoinGecko crypto data fetched for {ticker}")
                    return df, ticker
    except Exception as e:
        st.warning(f"⚠️ CoinGecko failed: {e}")

    st.error(f"❌ Could not fetch any data for {ticker}. Try again later.")
    st.stop()

def get_sentiment_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    if USE_TRANSFORMERS:
        try:
            out = sentiment_pipe(text[:512])
            label, score = out[0]['label'], out[0]['score']
            return score if label.upper().startswith('POS') else -score
        except Exception:
            return 0.0
    else:
        return vader.polarity_scores(text)['compound']

def fetch_reddit_posts(query, limit=200):
    try:
        posts = reddit.subreddit("stocks").search(query, limit=limit)
        data = [{"date": pd.to_datetime(p.created_utc, unit='s'),
                 "content": p.title + " " + (p.selftext or "")} for p in posts]
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"⚠️ Reddit fetch failed: {e}")
        return pd.DataFrame()

def fetch_google_news(query, start_date, end_date):
    gn = GoogleNews(start=str(start_date), end=str(end_date))
    gn.search(query)
    res = gn.result()
    df = pd.DataFrame(res)
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["content"] = df["title"].fillna('') + " " + df["desc"].fillna('')
    return df[["date", "content"]]

def backtest_predictions(df):
    df = df.copy()
    df["signal"] = df["label_pred"].shift(1).fillna(0)
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["strategy_returns"] = df["signal"] * df["returns"]
    df["cumulative_market"] = (1 + df["returns"]).cumprod()
    df["cumulative_strategy"] = (1 + df["strategy_returns"]).cumprod()
    return df

# ---------------------------
# Data Fetch
# ---------------------------
if st.button("🔄 Fetch & Prepare Data"):
    reddit_df = fetch_reddit_posts(stock_ticker, limit=max_posts)
    news_df = fetch_google_news(stock_ticker, start_date, end_date)
    combined = pd.concat([reddit_df, news_df]).dropna(subset=["content"]).reset_index(drop=True)

    if combined.empty:
        st.warning("No text data found for that query.")
    else:
        st.success(f"Collected {len(combined)} posts/headlines.")
        combined["sentiment"] = combined["content"].apply(get_sentiment_score)
        combined["date"] = pd.to_datetime(combined["date"]).dt.date
        daily_sent = combined.groupby("date")["sentiment"].mean().reset_index().rename(columns={"sentiment": "avg_sentiment"})

        ticker = auto_correct_ticker(stock_ticker)
        df, used_ticker = safe_yf_download(ticker, start_date, end_date)
        df = df.reset_index()
        if "Date" not in df.columns:
            df["Date"] = pd.to_datetime(df.index).date
        if "Close" not in df.columns:
            st.error("No 'Close' column found.")
            st.stop()
        if "Volume" not in df.columns:
            df["Volume"] = 0

        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        merged = pd.merge(df, daily_sent, left_on="Date", right_on="date", how="left")
        merged["avg_sentiment"] = merged["avg_sentiment"].fillna(0)
        merged["pct_change"] = merged["Close"].pct_change().fillna(0)
        merged["next_close"] = merged["Close"].shift(-1)
        merged["label"] = (merged["next_close"] > merged["Close"]).astype(int)
        merged = merged.dropna(subset=["label"])
        st.session_state["merged_df"] = merged
        st.write(merged[["Date", "Close", "avg_sentiment", "pct_change", "label"]].head())

# ---------------------------
# Model Training
# ---------------------------
if st.button("🧠 Train Model"):
    if "merged_df" not in st.session_state:
        st.error("Fetch data first.")
    else:
        merged = st.session_state["merged_df"]
        if "Volume" not in merged.columns:
            merged["Volume"] = 0
        X = merged[["avg_sentiment", "pct_change", "Volume"]].fillna(0)
        X["vol_scaled"] = X["Volume"] / (X["Volume"].max() + 1)
        X = X[["avg_sentiment", "pct_change", "vol_scaled"]]
        y = merged["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.session_state.update({"model": model, "X_test": X_test, "y_test": y_test, "y_pred": y_pred, "merged_df": merged})
        st.text(classification_report(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

        st.subheader("Feature Importance (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
            st.pyplot(bbox_inches="tight")
        except Exception as e:
            st.warning(f"SHAP skipped: {e}")

        merged["label_pred"] = model.predict(X)
        bt = backtest_predictions(merged)
        st.plotly_chart(px.line(bt, x="Date", y=["cumulative_market", "cumulative_strategy"],
                                title="Market vs Strategy Returns"))

# ---------------------------
# Live Prediction
# ---------------------------
st.header("Live Prediction")
input_date = st.date_input("Prediction Date", value=datetime.date.today())

if st.button("Predict Movement"):
    if "model" not in st.session_state:
        st.error("Train model first.")
    else:
        text_df = fetch_reddit_posts(stock_ticker, limit=100)
        if text_df.empty:
            text_df = fetch_google_news(stock_ticker, input_date - datetime.timedelta(days=3), input_date)
        avg_sent = text_df["content"].apply(get_sentiment_score).mean() if not text_df.empty else 0

        ticker = auto_correct_ticker(stock_ticker)
        df, used = safe_yf_download(ticker, input_date - datetime.timedelta(days=5), input_date)
        df = df.reset_index()
        if "Close" not in df.columns: st.stop()
        if "Volume" not in df.columns: df["Volume"] = 0

        pct = df["Close"].pct_change().iloc[-1]
        vol_scaled = df["Volume"].iloc[-1] / (df["Volume"].max() + 1)
        X_pred = pd.DataFrame([{"avg_sentiment": avg_sent, "pct_change": pct, "vol_scaled": vol_scaled}])
        model = st.session_state["model"]
        pred = model.predict(X_pred)[0]
        proba = model.predict_proba(X_pred)[0]
        arrow = "🔺 Up" if pred == 1 else "🔻 Down"
        st.metric(f"Prediction for {stock_ticker}", arrow, f"Confidence Up:{proba[1]:.2f} Down:{proba[0]:.2f}")
        st.write(f"Average sentiment: {avg_sent:.4f}")

st.markdown("---")
st.write("✅ Final version — fully stable, multi-source.")
