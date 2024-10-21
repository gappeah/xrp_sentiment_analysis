import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Establish the base URL for scraping
finviz_url_base = 'https://finviz.com/quote.ashx?t='

# Function to fetch the HTML content from Nike's website
def get_data(url):
    r = requests.get(url)
    r.raise_for_status()  # Ensure the request was successful
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup

# Function to parse product details from Nike's HTML content
def parse(soup):
    results = soup.find_all('div', {'class': 'product-card__body'})
    productlist = []

    for item in results:
        title_elem = item.find('div', {'class': 'product-card__title'})
        price_elem = item.find('div', {'class': 'product-price'})
        link_elem = item.find('a', {'class': 'product-card__link-overlay'})

        if title_elem and price_elem and link_elem:
            title = title_elem.text.strip()
            price = price_elem.text.strip().replace('£', '').replace(',', '')
            link = 'https://www.nike.com' + link_elem['href']

            product = {
                'Name': title,
                'Price': float(price) if price else 0.0,  # Convert price to float
                'Link': link  # Moved link after price
            }
            productlist.append(product)
            print(product)
    
    return productlist

# Function to output the scraped data to a CSV
def output(productlist):
    productsdf = pd.DataFrame(productlist)
    csv_path = "Price_BS4.csv"
    productsdf.to_csv(csv_path, index=False)  # Adjust the path
    print(f'Saved to CSV at {csv_path}')
    return productsdf

# Function to scrape multiple pages from Nike
def search_nike():
    product_name = input("Enter the product name to search on Nike: ")
    total_pages = int(input("Enter the number of pages to scrape: "))

    all_productlist = []

    for page in range(1, total_pages + 1):
        print(f"Scraping page {page}...")
        url = generate_nike_url(product_name, page)
        soup = get_data(url)
        productlist = parse(soup)
        all_productlist.extend(productlist)

    productsdf = output(all_productlist)
    return productsdf

# Main function to run the scraping and data visualization process
def main():
    productsdf = search_nike()

    # Load the CSV data for visualization
    df = pd.read_csv("Price_BS4.csv")
    
    # Optional: Clean the price data
    df['Price_Clean'] = df['Price']

    # Visualizations for the scraped data
    sns.set(style='whitegrid')

    # Bar plot of product prices
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Name', y='Price_Clean', data=df)
    plt.xticks(rotation=90)
    plt.title('Prices of Products')
    plt.xlabel('Product Name')
    plt.ylabel('Price (£)')
    plt.tight_layout()
    plt.show()

    # Box plot of prices by product name (or category if available)
    # You can replace 'Name' with a 'Category' field if your dataset has that.
    plt.figure(figsize=(50, 6))
    sns.boxplot(x='Name', y='Price_Clean', data=df)
    plt.title('Price Distribution by Product')
    plt.xlabel('Product Name')
    plt.ylabel('Price (£)')
    plt.xticks(rotation=90)  # Adjust fontsize here
    # Adjust layout to give more space to the x-axis labels
    plt.tight_layout()

    plt.show()

    # Histogram of price distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Price_Clean'], kde=True)
    plt.title('Price Distribution')
    plt.xlabel('Price (£)')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    main()