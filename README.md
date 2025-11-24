# Product Value Insights – ML Pipeline for EDA, Prediction & Clustering

This project analyzes retail product data and builds a complete machine-learning workflow that includes Exploratory Data Analysis (EDA), feature engineering, scaling, linear regression for prediction, and K-Means clustering for segmentation. The codebase is structured as a clean, modular ML pipeline rather than a single messy script or notebook.

## Features

- Automated dataset loading and validation  
- EDA: summary stats, missing value check, top product insights  
- Visualizations: price distribution, quantity comparison  
- Feature Engineering: TotalValue, Category Encoding, Standard Scaling  
- Linear Regression model for value prediction  
- Model evaluation using MSE and R²  
- K-Means clustering for product segmentation  
- Clear, production-ready file structure with reusable modules

## Project Structure

```
ml-product-analysis/
│
├── main.py
├── data/
│   └── product_dataset_1000.csv
├── notebooks/
│   └── exploratory.ipynb
├── src/
│   ├── load_data.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── cluster.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup & Installation

### 1. Clone the repository
```
git clone  https://github.com/gowthammayakannan/product_value_insights.git
cd product_value_insights
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Add the dataset
Place your CSV file inside the `data/` folder:
```
data/product_dataset_1000.csv
```
(If the file is large, do NOT commit it to GitHub.)

### 4. Run the project
```
python main.py
```

## Workflow Overview

### EDA
- Prints dataset summary, dtypes, missing values  
- Price distribution plot  
- Top 20 products by quantity visualization  

### Preprocessing
- Adds TotalValue = Price × Quantity  
- Label-encodes Category  
- Standardizes Price, Quantity, TotalValue  

### Model Training
- Train/Test split  
- Linear Regression model for predicting TotalValue (scaled)

### Evaluation
- Mean Squared Error (MSE)  
- R-Squared Score (R²)

### Clustering
- K-Means clustering using scaled features  
- Scatter plot of Price vs TotalValue with cluster labels  

## Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

## Why This Project Is Useful

This project demonstrates the ability to clean, process, analyze, visualize, and model real-world retail product data while following a clean, modular ML pipeline architecture. It’s suitable for portfolio use, analytics practice, and ML experimentation.

## License

This project is released under the MIT License and can be freely used, modified, or distributed.
