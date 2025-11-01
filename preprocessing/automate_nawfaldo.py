import pandas as pd
from sklearn.preprocessing import StandardScaler

def automate_preprocessing(df: pd.DataFrame):
    # Menghapus data kosong
    df = df.dropna(how="all")

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Drop duplicates
    df = df.drop_duplicates()

    # Standardization numerical features
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64','int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Outlier Handling
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    # Binning Example
    if 'HouseAge' in df.columns:
        df['HouseAge_bin'] = pd.qcut(df['HouseAge'], q=3, labels=['young','medium','old'])
        
    return df, scaler
