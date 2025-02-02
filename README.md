# XRP Sentiment Analysis Model

## Introduction

This project is designed to perform sentiment analysis on news headlines related to XRP, a popular cryptocurrency. The goal is to analyze the sentiment of news articles from various sources to determine whether the overall sentiment is bullish, bearish, or neutral. This can be useful for traders, investors, and enthusiasts who want to gauge market sentiment and make informed decisions.

## About XRP

XRP is a digital asset and cryptocurrency that was created by Ripple Labs Inc. It is designed to facilitate fast, low-cost international payments and is often used by financial institutions for cross-border transactions. Unlike Bitcoin, which uses a proof-of-work consensus mechanism, XRP uses a unique consensus algorithm called the Ripple Protocol Consensus Algorithm (RPCA). This allows for faster transaction times and lower energy consumption compared to traditional blockchain systems.

### Infrastructure and Governance

XRP operates on a decentralized ledger known as the XRP Ledger (XRPL). The XRPL is maintained by a network of independent validators that confirm transactions and maintain the integrity of the ledger. Ripple Labs, the company behind XRP, plays a significant role in the development and promotion of the XRP ecosystem, but the ledger itself is decentralized and not controlled by any single entity.

### How Transactions Work

Transactions on the XRP Ledger are processed in a matter of seconds, with low transaction fees. When a user initiates a transaction, it is broadcast to the network of validators. These validators then work together to reach a consensus on the validity of the transaction. Once consensus is reached, the transaction is added to the ledger, and the balance of the involved accounts is updated.

### Example of XRP Transaction (Pseudo Code)

```python
# Pseudo code for an XRP transaction
def send_xrp(sender, receiver, amount):
    if sender.balance >= amount:
        sender.balance -= amount
        receiver.balance += amount
        return "Transaction successful"
    else:
        return "Insufficient balance"
```

## Project Description

This project scrapes news headlines related to XRP from multiple sources, including Crypto News, The Crypto Basic, and Yahoo Finance. It then performs sentiment analysis on these headlines using two popular sentiment analysis tools: TextBlob and VADER. The results are visualized using Matplotlib and Seaborn, and the data is saved to a CSV file for further analysis.

### How to Launch the Project

1. **Clone the Repository**: Start by cloning this repository to your local machine.
   ```bash
   git clone https://github.com/gappeah/xrp-sentiment-analysis.git
   ```

2. **Install Dependencies**: Ensure you have Python installed, then install the required dependencies using pip.
   ```bash
   pip install pandas numpy beautifulsoup4 requests textblob nltk vaderSentiment matplotlib seaborn
   ```

3. **Download NLTK Resources**: The project uses NLTK for text processing. You need to download the necessary NLTK resources.
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Run the Script**: Execute the script to scrape news headlines, perform sentiment analysis, and generate visualizations.
   ```bash
   python xrp_sentiment_analysis.py
   ```

5. **View Results**: The script will generate a CSV file (`crypto_headlines_sentiment_analysis.csv`) containing the sentiment analysis results. It will also display visualizations of the sentiment scores over time.

Here is a detailed breakdown of the **code differences** between the **new version** and the **old version** of the XRP Sentiment Analysis Model. The differences are categorized by functionality and highlighted with explanations.

---

## 1. **Date Handling and Time-Based Analysis**

### New Version
```python
# Function to scrape headlines and dates from Crypto News
def scrape_crypto_news():
    headlines = []
    dates = []
    for page in range(1, 15):
        url = f"https://crypto.news/page/{page}/?s=xrp"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for h3, date_div in zip(soup.find_all('h3'), soup.find_all('div', class_='search-result-loop__date')):
            headline = h3.get_text()
            raw_date = date_div.get_text(strip=True).split(" at ")[0]  # Remove time
            parsed_date = datetime.strptime(raw_date, "%B %d, %Y").strftime('%d/%m/%Y')
            headlines.append(headline)
            dates.append(parsed_date)
    return headlines, dates
```

### Old Version
```python
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
```

### Key Differences:
- **New Version**: Extracts **dates** alongside headlines and parses them into a consistent format (`DD/MM/YYYY`).
- **Old Version**: Only extracts headlines, with no date handling.

---

## 2. **Yahoo Finance Date Parsing**

### New Version
```python
# Function to scrape headlines and dates from Yahoo Finance
def scrape_yahoo_finance():
    yahoo_headlines = []
    dates = []
    url = "https://finance.yahoo.com/quote/XRP-USD/news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('h3', class_='Mb(5px)')
    times = soup.find_all('div', class_='publishing yf-1weyqlp')

    for article, time_div in zip(articles[:50], times[:50]):  # Limit to 20 articles
        headline = article.get_text()
        time_text = time_div.get_text(strip=True).split("•")[-1].strip()
        # Calculate date from "XX hours/days ago"
        today = datetime.today()
        if "hour" in time_text:
            hours_ago = int(re.search(r"\d+", time_text).group())
            article_date = today - timedelta(hours=hours_ago)
        elif "day" in time_text:
            days_ago = int(re.search(r"\d+", time_text).group())
            article_date = today - timedelta(days=days_ago)
        elif "week" in time_text:
            weeks_ago = int(re.search(r"\d+", time_text).group())
            article_date = today - timedelta(weeks=weeks_ago)
        elif "month" in time_text:
            months_ago = int(re.search(r"\d+", time_text).group())
            article_date = today - timedelta(days=months_ago * 30)
        else:
            article_date = today

        parsed_date = article_date.strftime('%d/%m/%Y')
        yahoo_headlines.append(headline)
        dates.append(parsed_date)

    return yahoo_headlines, dates
```

### Old Version
```python
# Function to scrape headlines from Yahoo Finance
def scrape_yahoo_finance():
    yahoo_headlines = []
    url = "https://finance.yahoo.com/quote/XRP-USD/news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for h3 in soup.find_all('h3', class_='Mb(5px)'):
        yahoo_headlines.append(h3.get_text())
    return yahoo_headlines
```

### Key Differences:
- **New Version**: Dynamically calculates the date of each article based on relative time descriptions (e.g., "2 hours ago").
- **Old Version**: Does not extract or handle dates.

---

## 3. **Sentiment Categorization**

### New Version
```python
# Sentiment categorization in the new version
sentiment_category = (
    "Bullish" if average_score >= 0.5 else
    "Slightly Bullish" if 0.2 < average_score < 0.4 else
    "Neutral" if -0.2 <= average_score <= 0.2 else
    "Slightly Bearish" if -0.5 <= average_score < -0.2 else
    "Bearish"
)
```

### Old Version
```python
# Sentiment categorization in the old version
sentiment_category = (
    "Bullish" if average_score > 0.5 else
    "Slightly Bullish" if 0.2 < average_score <= 0.5 else
    "Neutral" if -0.2 <= average_score <= 0.2 else
    "Slightly Bearish" if -0.5 <= average_score < -0.2 else
    "Bearish"
)
```

### Key Differences:
- **New Version**: Uses a more granular categorization with clearer boundaries (e.g., `>= 0.5` for "Bullish").
- **Old Version**: Uses slightly different boundaries (e.g., `> 0.5` for "Bullish").

---

## 4. **Visualization**

### New Version
```python
# Visualization in the new version
fig, ax = plt.subplots(figsize=(15, 8))

sns.barplot(x="date", y="vader_score", data=df, color="blue", label="Vader Score", ax=ax)
sns.barplot(x="date", y="textblob_score", data=df, color="orange", label="TextBlob Score", ax=ax)

ax.set_title("Sentiment Analysis of Cryptocurrency Headlines")
ax.set_xlabel("Dates (DD/MM/YYYY)")
ax.set_ylabel("Sentiment Score")
ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for clarity
ax.legend()

plt.tight_layout()
plt.show()
```

### Old Version
```python
# Visualization in the old version
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x="headline", y="vader_score", data=df, label="Vader Score", ax=ax)
sns.barplot(x="headline", y="textblob_score", data=df, label="TextBlob Score", ax=ax)

ax.set_title("Sentiment Analysis of Cryptocurrency Headlines")
ax.set_xlabel("Headlines")
ax.set_ylabel("Sentiment Score")
ax.legend()
```

### Key Differences:
- **New Version**: Visualizes sentiment scores **over time** with dates on the x-axis.
- **Old Version**: Visualizes sentiment scores per headline, which is less informative.

---

## 5. **Data Storage**

### New Version
```python
# Data storage in the new version
data.append({
    'headline': headline,
    'date': date,
    'textblob_score': textblob_score,
    'vader_score': vader_score,
    'average_score': average_score,
    'sentiment_category': sentiment_category
})
```

### Old Version
```python
# Data storage in the old version
data.append({
    'headline': headline,
    'cleaned_headline': cleaned_headline,
    'textblob_score': textblob_score,
    'vader_score': vader_score,
    'average_score': average_score,
    'sentiment_category': sentiment_category
})
```

### Key Differences:
- **New Version**: Includes **dates** in the stored data.
- **Old Version**: Does not include dates.

---

## 6. **Descriptive Statistics**

### New Version
```python
# Descriptive statistics in the new version
def descriptive_statistics():
    print("Descriptive Statistics:\n")
    print(df[['textblob_score', 'vader_score', 'average_score']].describe())
```

### Old Version
```python
# Descriptive statistics in the old version
def descriptive_statistics(df):
    print("Descriptive Statistics of Sentiment Scores:")
    print(df[['textblob_score', 'vader_score', 'average_score']].describe())
```

### Key Differences:
- **New Version**: Simplified function call (no `df` parameter needed).
- **Old Version**: Requires `df` as a parameter.


### Visualizations
Between the old version and new version of the codebase improvement in the visualisation results in better quality graphs. 
The project includes several visualizations to help you understand the sentiment trends:

1. **Sentiment Scores Over Time**: A bar plot showing the sentiment scores (TextBlob and VADER) over time.
2. **TextBlob Sentiment Distribution**: A histogram showing the distribution of TextBlob sentiment scores.
3. **VADER Sentiment Distribution**: A histogram showing the distribution of VADER sentiment scores.
4. **TextBlob vs VADER Sentiment Scores**: A scatter plot comparing TextBlob and VADER sentiment scores.

### Old Version
![Figure_3](https://github.com/user-attachments/assets/4bd6b46b-2b89-43d6-9200-7077c3cf2147)
![Figure_2](https://github.com/user-attachments/assets/4567eee1-3ebb-41c3-8800-6f1ec40be8a0)
![Figure_1](https://github.com/user-attachments/assets/50645ed0-e4e4-4bcd-848e-9808f44a5002)
![Figure_4](https://github.com/user-attachments/assets/91b25b52-6f09-46e3-a1d8-e6feade6a0e5)

### New Version
![Figure_3_Revised](https://github.com/user-attachments/assets/713ee340-e33a-4c6e-b7de-34dddfab540b)
![Figure_2_Revised](https://github.com/user-attachments/assets/3204cab1-e0db-4efc-bd58-2706f2a6dd76)
![Figure_1_Revised](https://github.com/user-attachments/assets/1b1efcba-c118-447e-8287-715eefc5a75a)
![Figure_4_Revised](https://github.com/user-attachments/assets/bc81c202-face-4d93-8080-283dc7e15d65)

---

## Summary of Code Differences

| Feature                     | New Version Code Changes                                                                 | Old Version Code Changes                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| **Date Handling**            | Extracts and parses dates from all sources.                                             | No date handling.                                                                       |
| **Yahoo Finance Parsing**    | Dynamically calculates dates from relative time descriptions.                           | No date extraction.                                                                     |
| **Sentiment Categorization** | More granular and clear boundaries for sentiment categories.                            | Simpler categorization with less granularity.                                           |
| **Visualization**            | Time-based visualizations with dates on the x-axis.                                     | Headline-based visualizations without time context.                                     |
| **Data Storage**             | Includes dates in the stored data.                                                      | Does not include dates.                                                                 |
| **Descriptive Statistics**   | Simplified function call.                                                               | Requires `df` as a parameter.                                                           |

---

## Conclusion

This project provides a comprehensive analysis of sentiment trends in XRP-related news headlines. By combining web scraping, sentiment analysis, and data visualization, it offers valuable insights into market sentiment, which can be used to inform trading and investment decisions.


