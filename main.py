import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import plotly.express as px
import pandas as pd

# Step 1: Scrape the cryptocurrency headlines
url = "https://crypto.news/?s=xrp"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract headlines from h3 tags
headlines = [h3.get_text() for h3 in soup.find_all("h3")]

# Step 2: Perform sentiment analysis using VADER and TextBlob
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

data = []
for headline in headlines:
    # VADER sentiment analysis
    vader_score = sia.polarity_scores(headline)["compound"]

    # TextBlob sentiment analysis
    textblob_score = TextBlob(headline).sentiment.polarity

    data.append(
        {
            "headline": headline,
            "vader_score": vader_score,
            "textblob_score": textblob_score,
        }
    )

# Step 3: Create a DataFrame
df = pd.DataFrame(data)

# Step 4: Plot the results using Plotly
fig = px.bar(
    df,
    x="headline",
    y=["vader_score", "textblob_score"],
    title="Sentiment Analysis of Cryptocurrency Headlines",
    labels={"value": "Sentiment Score", "variable": "Sentiment Method"},
    text="headline",
)

fig.update_traces(texttemplate="%{text}", textposition="outside")
fig.update_layout(
    barmode="group", xaxis_title="Headlines", yaxis_title="Sentiment Score"
)
fig.show()
