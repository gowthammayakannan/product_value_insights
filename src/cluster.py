import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def run_kmeans(df):
    X_scaled = df[['Price_scaled', 'Quantity_scaled', 'TotalValue_scaled']]
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='Price',
        y='TotalValue',
        hue='Cluster',
        data=df,
        palette='Set2'
    )
    plt.title("K-Means Clustering of Products")
    plt.show()

    return df
