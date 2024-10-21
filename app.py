import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Establish the base URL for scraping
finviz_url_base = 'https://finviz.com/quote.ashx?t='

tickers = ['AMD ', 'META', 'AAPL', 'AMZN', 'GOOG', 'MSFT', 'NVDA', 'TSLA', 'SPY']

for ticker in tickers:
    url = finviz_url_base + ticker
    request = requests(url=url, header={'user-agent': 'my-app/0.0.1'})
    print(f'Getting data for {ticker}...')

    response = urlopen(request)
    print(response)
    
    
