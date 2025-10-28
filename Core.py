import pandas as pd
import os
import numpy as np 
import logging
from FileLoader import smart_load

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_and_convert_series(s: pd.Series) -> pd.Series:
    """
    Infers the best data type for a pandas Series and converts it.
    """
    #if already numeric/datetime leave it alone
    if not pd.api.types.is_object_dtype(s):
        return s
    
    # Drop NaN and empty strings for inference
    sample = s.dropna()
    if pd.api.types.is_string_dtype(sample) or pd.api.types.is_object_dtype(sample):
        sample = sample[sample.astype(str).str.strip() != '']
    
    if sample.empty:
        return s
    # Try datetime first
    try:
        pd.to_datetime(sample, errors='raise')
        return pd.to_datetime(s, errors='coerce')
    except Exception:
        pass

    # Try numeric next
    try:
        pd.to_numeric(sample, errors='raise')
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        pass

    # Try timedelta
    try:
        pd.to_timedelta(sample, errors='raise')
        return pd.to_timedelta(s, errors='coerce')
    except Exception:
        pass

    # normalize whitespace and return as stripped strings
    return s.astype(str).str.strip()
    

def clean_file(file_path):
    try:
        df = smart_load(file_path)

        # Ensure we have a DataFrame and work on a copy to avoid mutating cached objects
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"smart_load returned {type(df)}, expected pandas.DataFrame")
        df = df.copy()

        # remove exact duplicate rows
        df = df.drop_duplicates()

        # normalize column names (allow letters, numbers and underscores)
        df.columns = (df.columns
                      .str.strip()
                      .str.lower()
                      .str.replace(' ', "_")
                      .str.replace(r'[^0-9a-zA-Z_]', '', regex=True)
                      )

        # infer and convert dtypes per column for object columns
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = infer_and_convert_series(df[col])

        # fill missing values: numeric -> 0, datetime -> NaT, others -> 'unknown'
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(pd.NaT)
            else:
                df[col] = df[col].fillna('unknown')

        return df
    except Exception as e:
        logger.error(f"error in cleaning file {file_path}: {str(e)}")
        raise
    
def remove_duplicate_rows(row):
    """
    We should ask the user if we want to remove any duplicate rows 
    """
    #detects any duplicate rows
    duplicate_rows = row[row.duplicated()]
    #print for now to test
    print(duplicate_rows)

    #removes the duplicate row
    df = row.drop_duplicates()

    return df

def correlation_matrix_pearson(file_path):
    df = smart_load(file_path)

    #get the matrix
    corr_matrix = df.corr(method='pearson', min_periods=1, numeric_only=False).round(2)

    return corr_matrix

def correlation_matrix_kendall(file_path):
    df = smart_load(file_path)

    #get the matrix
    corr_matrix = df.corr(method='kendall', min_periods=1, numeric_only=False).round(2)

    return corr_matrix

def correlation_matrix_spearman(file_path):
    df = smart_load(file_path)

    #get the matrix
    corr_matrix = df.corr(method='spearman', min_periods=1, numeric_only=False).round(2)

    return corr_matrix

def find_similar_col_to_remove(file_path):
    df = smart_load(file_path)

    #calcualte correlation matrix
    corr_matrix = df.corr().abs()

    """triu is the upper triangle of the correlation matrix, just want the upper to prevent duplicate correlations
    np.ones inputs 1's into matrix
    corr_matrix.shape creates the matrix
    k=1 starts the matrix one diagonal above the main diagonal(the bottom row to prevent duplicates so upper values are taken only)
    astype bool sets the values to true or false 
    """
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # this loops through the upper columns to detect any high correlations and saves them
    to_drop = [column for column in upper.columns if (upper[column] > 0.95).any()]

    # drop the highly correlated columns and return the reduced DataFrame
    df = df.drop(columns=to_drop)
    return df




        
