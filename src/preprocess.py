from sklearn.preprocessing import LabelEncoder, StandardScaler

def add_total_value(df):
    df['TotalValue'] = df['Price'] * df['Quantity']
    return df

def encode_category(df):
    le = LabelEncoder()
    df['Category_Encoded'] = le.fit_transform(df['Category'])
    return df

def scale_features(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[['Price', 'Quantity', 'TotalValue']])
    df[['Price_scaled', 'Quantity_scaled', 'TotalValue_scaled']] = scaled
    return df
