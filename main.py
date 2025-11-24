from src.load_data import load_dataset
from src.eda import run_basic_eda, plot_price_distribution, plot_top_products
from src.preprocess import add_total_value, encode_category, scale_features
from src.train_model import train_linear_regression
from src.evaluate import evaluate_model
from src.cluster import run_kmeans

def main():
    df = load_dataset("data/product_dataset_1000.csv")

    # EDA
    run_basic_eda(df)
    plot_price_distribution(df)
    plot_top_products(df)

    # Preprocessing
    df = add_total_value(df)
    df = encode_category(df)
    df = scale_features(df)

    # Model Training
    model, X_test, y_test = train_linear_regression(df)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Clustering
    df = run_kmeans(df)

if __name__ == "__main__":
    main()
