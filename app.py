import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to scrape headlines from Crypto News
def scrape_crypto_news():
    headlines = []
    for page in range(1, 6):
        url = f"https://crypto.news/page/{page}/?s=xrp"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for h3 in soup.find_all('h3'):
            headlines.append(h3.get_text())
    return headlines

# Function to scrape titles from The Crypto Basic
def scrape_crypto_basic():
    titles = []
    for page in range(1, 6):
        url = f"https://thecryptobasic.com/tag/ripple/page/{page}/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for a in soup.find_all('a'):
            title = a.find('h3')
            if title:
                titles.append(title.get_text())
    return titles

# Function to scrape headlines from Yahoo Finance
def scrape_yahoo_finance():
    yahoo_headlines = []
    url = "https://finance.yahoo.com/quote/XRP-USD/news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for h3 in soup.find_all('h3', class_='Mb(5px)'):
        yahoo_headlines.append(h3.get_text())
    return yahoo_headlines

# Step 2: Combine all scraped data
def combine_data():
    crypto_news_headlines = scrape_crypto_news()
    crypto_basic_titles = scrape_crypto_basic()
    yahoo_finance_headlines = scrape_yahoo_finance()

    all_headlines = crypto_news_headlines + crypto_basic_titles + yahoo_finance_headlines
    return all_headlines

# Step 3: Perform sentiment analysis using TextBlob
def analyze_sentiment_textblob(headline):
    analysis = TextBlob(headline)
    return analysis.sentiment.polarity

# Step 4: Perform sentiment analysis using VADER
def analyze_sentiment_vader(headline):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(headline)
    return score['compound']

# Step 5: Clean and preprocess the headlines
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Step 6: Create a DataFrame and analyze sentiment
data = []
all_headlines = combine_data()
for headline in all_headlines:
    cleaned_headline = clean_text(headline)
    textblob_score = analyze_sentiment_textblob(cleaned_headline)
    vader_score = analyze_sentiment_vader(cleaned_headline)
    average_score = (textblob_score + vader_score) / 2
    sentiment_category = (
        "Bullish" if average_score > 0.5 else
        "Slightly Bullish" if 0.2 < average_score <= 0.5 else
        "Neutral" if -0.2 <= average_score <= 0.2 else
        "Slightly Bearish" if -0.5 <= average_score < -0.2 else
        "Bearish"
    )
    data.append({
        'headline': headline,
        'cleaned_headline': cleaned_headline,
        'textblob_score': textblob_score,
        'vader_score': vader_score,
        'average_score': average_score,
        'sentiment_category': sentiment_category
    })

df = pd.DataFrame(data)

# Step 7: Display the DataFrame
print(df)

# Step 8: Save the DataFrame to a CSV file
df.to_csv('crypto_headlines_sentiment_analysis.csv', index=False)

# Step 9: Descriptive Statistics
def descriptive_statistics(df):
    print("Descriptive Statistics of Sentiment Scores:")
    print(df[['textblob_score', 'vader_score', 'average_score']].describe())

# Step 10: Visualize Sentiment Scores
def visualize_sentiment(df):
    sns.set(style="whitegrid")

    # TextBlob Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['textblob_score'], bins=30, kde=True, color='blue')
    plt.title('Distribution of TextBlob Sentiment Scores')
    plt.xlabel('TextBlob Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

    # VADER Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['vader_score'], bins=30, kde=True, color='green')
    plt.title('Distribution of VADER Sentiment Scores')
    plt.xlabel('VADER Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

    # TextBlob vs VADER Sentiment Scores
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='textblob_score', y='vader_score', data=df, hue='sentiment_category', palette='coolwarm')
    plt.title('TextBlob vs VADER Sentiment Scores')
    plt.xlabel('TextBlob Sentiment Score')
    plt.ylabel('VADER Sentiment Score')
    plt.axhline(0, color='red', linestyle='--')
    plt.axvline(0, color='blue', linestyle='--')
    plt.show()
    
    # Step 4: Plot the results using Seaborn
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the results using seaborn
sns.barplot(x="headline", y="vader_score", data=df, label="Vader Score", ax=ax)
sns.barplot(x="headline", y="textblob_score", data=df, label="TextBlob Score", ax=ax)

# Set title and labels
ax.set_title("Sentiment Analysis of Cryptocurrency Headlines")
ax.set_xlabel("Headlines")
ax.set_ylabel("Sentiment Score")

# Add legend
ax.legend()

# Step 11: Execute the functions
descriptive_statistics(df)
visualize_sentiment(df)

# Step 12: Print Summary of Sentiment
sentiment_summary = df['sentiment_category'].value_counts()
print("\nSentiment Summary:")
print(sentiment_summary)
