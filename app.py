import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple


class XRPSentimentAnalyzer:
    def __init__(self):
        """Initialize the XRP Sentiment Analyzer with necessary NLTK downloads and custom lexicon."""
        # Download required NLTK data
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")

        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Custom crypto-specific sentiment lexicon
        self.crypto_lexicon = {
            "positive": ["bullish", "adoption", "partnership", "breakthrough", "surge"],
            "negative": ["bearish", "lawsuit", "dump", "fud", "regulatory"],
            "neutral": ["hold", "stable", "announcement", "update"],
        }

    def scrape_news(self, days_back: int = 30) -> List[Dict]:
        """
        Scrape XRP-related news from various sources.

        Args:
            days_back: Number of days of historical news to retrieve

        Returns:
            List of dictionaries containing news data
        """
        news_data = []
        sources = [
            "https://coinmarketcap.com/currencies/xrp/news/",
            "https://ripple.com/insights/",
        ]

        for source in sources:
            try:
                response = requests.get(source)
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract news titles and dates (implementation depends on specific site structure)
                # This is a simplified example
                articles = soup.find_all("article")
                for article in articles:
                    title = article.find("h2").text if article.find("h2") else ""
                    date_str = article.find("time").text if article.find("time") else ""

                    if title and date_str:
                        news_data.append(
                            {"title": title, "date": date_str, "source": source}
                        )

            except Exception as e:
                print(f"Error scraping {source}: {str(e)}")

        return news_data

    def get_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch XRP market data from Yahoo Finance.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with XRP price and volume data
        """
        xrp = yf.download("XRP-USD", start=start_date, end=end_date)
        xrp["Returns"] = xrp["Close"].pct_change() # Replace "Adj Close" with "Close"
        xrp["Volatility"] = xrp["Returns"].rolling(window=20).std()
        return xrp

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text data.

        Args:
            text: Raw text string

        Returns:
            Preprocessed text string
        """
        # Conv ert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove special characters and numbers
        text = re.sub(r"[^\w\s]", "", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]

        return " ".join(tokens)

    def calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment score using both TextBlob and custom crypto lexicon.

        Args:
            text: Preprocessed text string

        Returns:
            Combined sentiment score
        """
        # TextBlob sentiment
        blob_sentiment = TextBlob(text).sentiment.polarity

        # Custom lexicon sentiment
        words = set(text.split())
        positive_matches = len(words.intersection(self.crypto_lexicon["positive"]))
        negative_matches = len(words.intersection(self.crypto_lexicon["negative"]))
        custom_sentiment = (positive_matches - negative_matches) / (
            positive_matches + negative_matches + 1
        )

        # Combine both sentiment scores (giving more weight to crypto-specific lexicon)
        return 0.3 * blob_sentiment + 0.7 * custom_sentiment

    def create_visualizations(
        self, sentiment_df: pd.DataFrame, market_df: pd.DataFrame
    ) -> None:
        """
        Create interactive visualizations using Plotly.

        Args:
            sentiment_df: DataFrame with sentiment analysis results
            market_df: DataFrame with market data
        """
        # Create sentiment over time visualization
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=sentiment_df.index,
                y=sentiment_df["sentiment_score"],
                name="Sentiment Score",
                line=dict(color="blue"),
            )
        )
        fig1.update_layout(
            title="XRP Sentiment Score Over Time",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
        )
        fig1.show()

        # Create price and sentiment correlation visualization
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=market_df.index,
                y=market_df["Adj Close"],
                name="XRP Price",
                line=dict(color="green"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=sentiment_df.index,
                y=sentiment_df["sentiment_score"] * market_df["Adj Close"].mean(),
                name="Normalized Sentiment",
                line=dict(color="red"),
            )
        )
        fig2.update_layout(
            title="XRP Price vs Sentiment",
            xaxis_title="Date",
            yaxis_title="Price / Normalized Sentiment",
        )
        fig2.show()


def main():
    # Initialize analyzer
    analyzer = XRPSentimentAnalyzer()

    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Collect data
    news_data = analyzer.scrape_news(days_back=30)
    market_data = analyzer.get_market_data(
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )

    # Process news data
    sentiment_results = []
    for news in news_data:
        processed_text = analyzer.preprocess_text(news["title"])
        sentiment_score = analyzer.calculate_sentiment_score(processed_text)
        sentiment_results.append(
            {
                "date": news["date"],
                "title": news["title"],
                "sentiment_score": sentiment_score,
            }
        )

    # Create sentiment DataFrame
    sentiment_df = pd.DataFrame(sentiment_results)
    print(sentiment_df)
    
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]) # Error in  232
    
    
    
    sentiment_df.set_index("date", inplace=True)

    # Create visualizations
    analyzer.create_visualizations(sentiment_df, market_data)


if __name__ == "__main__":
    main()
