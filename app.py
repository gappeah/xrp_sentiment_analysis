# sentiment_analysis.py

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import datetime
from flask import Flask, render_template, request

# Sentiment Analysis
class SentimentAnalysis:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def clean_text(self, text):
        # Implement text cleaning (e.g., removing punctuation, lowercasing, etc.)
        cleaned_text = text.lower()
        # Add more cleaning steps as needed
        return cleaned_text

    def analyze_sentiment(self, text):
        cleaned_text = self.clean_text(text)
        sentiment_score = self.analyzer.polarity_scores(cleaned_text)
        return sentiment_score

    def classify_sentiment(self, score):
        # Classify sentiment based on the score (positive, neutral, negative)
        if score['compound'] >= 0.05:
            return 'positive'
        elif score['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'

# Stock Data
class StockData:
    def __init__(self, ticker):
        self.ticker = ticker

    def fetch_stock_prices(self, start_date, end_date):
        stock_data = yf.download(self.ticker, start=start_date, end=end_date)
        return stock_data

    def correlate_with_sentiment(self, stock_data, sentiment_data):
        # Implement correlation analysis between sentiment and stock prices
        merged_data = pd.merge(stock_data, sentiment_data, left_index=True, right_index=True)
        correlation = merged_data['Close'].corr(merged_data['sentiment_score'])
        return correlation

# Data Visualization
class DataVisualization:
    def __init__(self):
        pass

    def plot_time_series(self, stock_data, sentiment_data):
        fig = px.line(stock_data, x=stock_data.index, y='Close', title='Stock Prices Over Time')
        fig.show()

    def plot_sentiment_heatmap(self, sentiment_data):
        # Implement heatmap visualization
        pass

    def plot_candlestick_with_sentiment(self, stock_data, sentiment_data):
        fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                             open=stock_data['Open'],
                                             high=stock_data['High'],
                                             low=stock_data['Low'],
                                             close=stock_data['Close'])])
        # Overlay sentiment data
        fig.show()

# Web Interface
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Perform sentiment analysis and stock data correlation
    sentiment_analyzer = SentimentAnalysis()
    stock_data = StockData(ticker)
    visualizer = DataVisualization()

    # Example placeholder logic
    stock_prices = stock_data.fetch_stock_prices(start_date, end_date)
    # Replace with actual sentiment data
    sentiment_data = pd.DataFrame({'date': stock_prices.index, 'sentiment_score': np.random.randn(len(stock_prices))})
    correlation = stock_data.correlate_with_sentiment(stock_prices, sentiment_data)

    # Plot results
    visualizer.plot_time_series(stock_prices, sentiment_data)
    
    return f"Correlation between sentiment and stock prices: {correlation}"

if __name__ == '__main__':
    app.run(debug=True)

# machine_learning.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SentimentPredictionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        return mse, mae

    def predict(self, X):
        return self.model.predict(X)
