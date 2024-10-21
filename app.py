import requests
from bs4 import BeautifulSoup


# Establish the base URL for scraping
finviz_url_base = 'https://finviz.com/quote.ashx?t='

tickers = ['AMD', 'META', 'AAPL', 'AMZN', 'GOOG', 'MSFT', 'NVDA', 'TSLA', 'SPY']

news_table = {}

for ticker in tickers:
    url = finviz_url_base + ticker
    request = requests.get(url=url, headers={'user-agent': 'my-app/0.0.1'})  # Use requests.get with headers
    print(f'Getting data for {ticker}...')

    # The request object holds the response, so you can parse it with BeautifulSoup.
    soup = BeautifulSoup(request.text, 'html.parser')
    news_table = soup.find(id='news-table')
    news_tables[ticker] = news_table
    break
print(news_tables)  # Visualize the response (HTML) from Finviz

    
    
