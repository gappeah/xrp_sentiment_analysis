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
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Data class for storing news article information"""

    title: str
    date: str
    source: str
    sentiment_score: Optional[float] = None


class XRPSentimentAnalyzer:
    def __init__(self):
        """Initialize the XRP Sentiment Analyzer with necessary NLTK downloads and custom lexicon."""
        self._initialize_nltk()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self._initialize_crypto_lexicon()

    def _initialize_nltk(self) -> None:
        """Download required NLTK data with error handling"""
        required_nltk_data = ["punkt", "stopwords", "wordnet"]
        for data in required_nltk_data:
            try:
                nltk.download(data, quiet=True)
            except Exception as e:
                logger.error(f"Failed to download NLTK data {data}: {str(e)}")
                raise

    def _initialize_crypto_lexicon(self) -> None:
        """Initialize the crypto-specific sentiment lexicon"""
        self.crypto_lexicon = {
            "positive": [
                "bullish",
                "adoption",
                "partnership",
                "breakthrough",
                "surge",
                "rally",
                "gain",
                "growth",
                "innovation",
                "success",
            ],
            "negative": [
                "bearish",
                "lawsuit",
                "dump",
                "fud",
                "regulatory",
                "crash",
                "decline",
                "risk",
                "warning",
                "ban",
            ],
            "neutral": [
                "hold",
                "stable",
                "announcement",
                "update",
                "development",
                "news",
                "report",
                "analysis",
                "review",
                "status",
            ],
        }

    def scrape_news(self, days_back: int = 30) -> List[NewsArticle]:
        """
        Scrape XRP-related news with improved HTML parsing and error handling.

        Args:
            days_back: Number of days of historical news to retrieve

        Returns:
            List of NewsArticle objects
        """
        news_data = []
        sources = ["https://crypto.news/?s=xrp"]

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        def scrape_single_source(source: str) -> List[NewsArticle]:
            try:
                response = requests.get(source, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Find all news articles within div containers
                articles = soup.find_all("div", class_="article-container")
                source_news = []

                for article in articles:
                    # Extract title from h3 tag
                    title_tag = article.find("h3", class_="article-title")
                    if not title_tag:
                        continue

                    title = title_tag.text.strip()

                    # Extract date with better parsing
                    date_tag = article.find("time", class_="article-date")
                    date_str = date_tag.get("datetime") if date_tag else ""

                    if title and date_str:
                        source_news.append(
                            NewsArticle(title=title, date=date_str, source=source)
                        )

                return source_news

            except requests.RequestException as e:
                logger.error(f"Error scraping {source}: {str(e)}")
                return []

        # Use ThreadPoolExecutor for parallel scraping
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(scrape_single_source, sources))

        # Flatten results
        for result in results:
            news_data.extend(result)

        return news_data

    def get_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch XRP market data with improved error handling and calculations.
        """
        try:
            xrp = yf.download("XRP-USD", start=start_date, end=end_date)

            # Calculate returns and volatility
            xrp["Returns"] = xrp["Close"].pct_change()
            xrp["Volatility"] = xrp["Returns"].rolling(window=20, min_periods=1).std()

            # Add technical indicators
            xrp["SMA_20"] = xrp["Close"].rolling(window=20, min_periods=1).mean()
            xrp["RSI"] = self._calculate_rsi(xrp["Close"])

            return xrp

        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def preprocess_text(self, text: str) -> str:
        """
        Enhanced text preprocessing with better regex patterns and error handling.
        """
        try:
            # Convert to lowercase and remove URLs
            text = text.lower()
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

            # Remove special characters while preserving important punctuation
            text = re.sub(r"[^\w\s\-\.]", "", text)

            # Tokenize and remove stopwords
            tokens = word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 2
            ]

            return " ".join(tokens)

        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return text

    def calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment with improved weighting and error handling.
        """
        try:
            # TextBlob sentiment
            blob_sentiment = TextBlob(text).sentiment.polarity

            # Custom lexicon sentiment with improved calculation
            words = set(text.split())
            positive_matches = len(words.intersection(self.crypto_lexicon["positive"]))
            negative_matches = len(words.intersection(self.crypto_lexicon["negative"]))
            neutral_matches = len(words.intersection(self.crypto_lexicon["neutral"]))

            if positive_matches + negative_matches + neutral_matches == 0:
                return blob_sentiment

            custom_sentiment = (positive_matches - negative_matches) / (
                positive_matches + negative_matches + neutral_matches
            )

            # Weighted combination
            return 0.4 * blob_sentiment + 0.6 * custom_sentiment

        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            return 0.0

    def create_visualizations(
        self, sentiment_df: pd.DataFrame, market_df: pd.DataFrame
    ) -> None:
        """
        Create enhanced interactive visualizations with better styling and features.
        """
        try:
            # Sentiment over time
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    x=sentiment_df.index,
                    y=sentiment_df["sentiment_score"],
                    name="Sentiment Score",
                    line=dict(color="blue", width=2),
                    mode="lines+markers",
                )
            )

            fig1.update_layout(
                title="XRP Sentiment Analysis",
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                hovermode="x unified",
            )

            # Price and sentiment correlation
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=market_df.index,
                    y=market_df["Close"],
                    name="XRP Price",
                    line=dict(color="green", width=2),
                )
            )

            # Normalize sentiment to price scale
            norm_factor = (
                market_df["Close"].mean() / sentiment_df["sentiment_score"].mean()
            )
            fig2.add_trace(
                go.Scatter(
                    x=sentiment_df.index,
                    y=sentiment_df["sentiment_score"] * norm_factor,
                    name="Normalized Sentiment",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

            fig2.update_layout(
                title="XRP Price vs Sentiment Correlation",
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified",
            )

            fig1.show()
            fig2.show()

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")


def main():
    try:
        analyzer = XRPSentimentAnalyzer()

        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Get news data
        news_data = analyzer.scrape_news(days_back=30)

        # Process news and calculate sentiment
        sentiment_results = []
        for article in news_data:
            processed_text = analyzer.preprocess_text(article.title)
            sentiment_score = analyzer.calculate_sentiment_score(processed_text)
            sentiment_results.append(
                {
                    "date": article.date,
                    "title": article.title,
                    "sentiment_score": sentiment_score,
                }
            )

        # Create and process DataFrames
        sentiment_df = pd.DataFrame(sentiment_results)
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        sentiment_df.set_index("date", inplace=True)

        # Get market data
        market_data = analyzer.get_market_data(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        # Create visualizations
        analyzer.create_visualizations(sentiment_df, market_data)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
