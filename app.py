import requests  # Using requests instead of urllib
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Finviz base URL
finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'FB']

# Dictionary to store news tables for each ticker
news_tables = {}

# Loop through each ticker to fetch data
for ticker in tickers:
    url = finviz_url + ticker
    response = requests.get(url=url, headers={'user-agent': 'my-app/0.0.1'})  # Fetch the page content
    soup = BeautifulSoup(response.text, 'html.parser')  # Parse the content with BeautifulSoup
    news_table = soup.find(id='news-table')  # Find the news table by its ID
    
    if news_table:
        news_tables[ticker] = news_table
    else:
        print(f'No news table found for {ticker}')
    
# List to hold parsed data
parsed_data = []

# Loop through each ticker's news table
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text  # Extract the title of the news article
        date_data = row.td.text.split(' ')  # Extract the date and time
        
        if len(date_data) == 1:
            time = date_data[0]
            date = None  # No date means it's from the same day as previous entries
        else:
            date = date_data[0]
            time = date_data[1]
        
        parsed_data.append([ticker, date, time, title])

# Create a DataFrame from the parsed data
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

# Initialize the VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Function to get the compound sentiment score for each title
def get_compound_score(title):
    return vader.polarity_scores(title)['compound']

# Apply sentiment analysis to the titles
df['compound'] = df['title'].apply(get_compound_score)

# Convert the date column to datetime format, ignoring rows without dates
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

# Plotting sentiment over time
plt.figure(figsize=(10, 8))

# Calculate the mean sentiment for each ticker and date
mean_df = df.groupby(['ticker', 'date']).mean().unstack()

# Extract the compound sentiment scores
mean_df = mean_df.xs('compound', axis="columns")

# Plot the data as a bar plot
mean_df.plot(kind='bar', title="Mean Sentiment by Ticker Over Time")
plt.show()
