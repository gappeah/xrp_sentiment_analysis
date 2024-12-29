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

# Step 1: Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to scrape headlines from Crypto News
def scrape_crypto_news():
    headlines = []
    for page in range(1, 6):  # Cycle through 5 pages
        url = f"https://crypto.news/page/{page}/?s=xrp"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for h3 in soup.find_all('h3'):
            headlines.append(h3.get_text())
    return headlines

# Function to scrape titles from The Crypto Basic
def scrape_crypto_basic():
    titles = []
    for page in range(1, 6):  # Cycle through 5 pages
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
def analyze_sentiment(headline):
    analysis = TextBlob(headline)
    return analysis.sentiment.polarity

# Step 4: Clean and preprocess the headlines
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Step 5: Create a DataFrame and analyze sentiment
data = []
all_headlines = combine_data()
for headline in all_headlines:
    cleaned_headline = clean_text(headline)
    sentiment_score = analyze_sentiment(cleaned_headline)
    data.append({
        'headline': headline,
        'cleaned_headline': cleaned_headline,
        'sentiment_score': sentiment_score
    })

df = pd.DataFrame(data)

# Step 6: Display the DataFrame
print(df)

# Optional: Save the DataFrame to a CSV file
df.to_csv('crypto_headlines_sentiment_analysis.csv', index=False)


import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Descriptive Statistics
def descriptive_statistics(df):
    print("Descriptive Statistics of Sentiment Scores:")
    print(df['sentiment_score'].describe())

# Step 2: Visualize Sentiment Scores
def visualize_sentiment(df):
    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Histogram of sentiment scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment_score'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.axvline(df['sentiment_score'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(df['sentiment_score'].median(), color='yellow', linestyle='dashed', linewidth=1)
    plt.legend({'Mean': df['sentiment_score'].mean(), 'Median': df['sentiment_score'].median()})
    plt.show()

    # Bar chart of sentiment score counts
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment_score'].value_counts(bins=10).sort_index()
    sentiment_counts.plot(kind='bar', color='orange')
    plt.title('Count of Sentiment Scores Binned')
    plt.xlabel('Sentiment Score Bins')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Execute the functions
descriptive_statistics(df)
visualize_sentiment(df)