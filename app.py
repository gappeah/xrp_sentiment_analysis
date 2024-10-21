import requests
from bs4 import BeautifulSoup

# Base URL for scraping
finviz_url_base = 'https://finviz.com/quote.ashx?t='

# List of tickers to scrape
tickers = ['AMD', 'META', 'AAPL', 'AMZN', 'GOOG', 'MSFT', 'NVDA', 'TSLA', 'SPY']

# Dictionary to hold news tables for each ticker
news_tables = {}

for ticker in tickers:
    url = finviz_url_base + ticker
    request = requests.get(url=url, headers={'user-agent': 'my-app/0.0.1'})
    print(f'Getting data for {ticker}...')

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(request.text, 'html.parser')

    # Find the news table using its ID
    news_table = soup.find(id='news-table')
    
    # Store the news table HTML in the dictionary, with the ticker as the key
    if news_table:
        news_tables[ticker] = news_table
    else:
        print(f'No news table found for {ticker}')
    
    break  # Just to test the first ticker, remove or modify this for all tickers

# Visualize the result
print(news_tables)
