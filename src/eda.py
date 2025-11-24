import matplotlib.pyplot as plt
import seaborn as sns

def run_basic_eda(df):
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.dtypes)
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())

def plot_price_distribution(df):
    sns.histplot(df['Price'], kde=True)
    plt.title('Distribution of Price')
    plt.show()

def plot_top_products(df):
    sns.barplot(x='ProductName', y='Quantity', data=df.head(20))
    plt.xticks(rotation=90)
    plt.title('Top 20 Products by Quantity')
    plt.show()
