import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from datetime import datetime


def automate_preprocessing(
    df: pd.DataFrame,
    save: bool = True,
    save_dir: str = "preprocessing/california_preprocessing",
    dataset_name: str = "california-housing_preprocessing",
    timestamp: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler, Optional[str]]:
    # Drop rows that are fully empty
    df = df.dropna(how="all").copy()

    # Impute missing values for all columns
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    # Drop duplicates
    df = df.drop_duplicates()

    # Standardization
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_columns) > 0:
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Outlier handling
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    # Binning
    if "HouseAge" in df.columns:
        try:
            df["HouseAge_bin"] = pd.qcut(df["HouseAge"], q=3, labels=["young", "medium", "old"])
        except Exception:
            df["HouseAge_bin"] = pd.cut(df["HouseAge"], bins=3, labels=["young", "medium", "old"])

    # Ensure save directory exists
    saved_path = None
    if save:
        os.makedirs(save_dir, exist_ok=True)
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{dataset_name}_{ts}.csv"
        else:
            filename = f"{dataset_name}.csv"
        saved_path = os.path.join(save_dir, filename)
        df.to_csv(saved_path, index=False)

    return df, scaler, saved_path
