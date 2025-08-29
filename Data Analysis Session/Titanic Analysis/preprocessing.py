import pandas as pd
import numpy as np

def check_data_types(df):
    """
    Check the data types of all columns in a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame to check
        
    Returns:
        pandas.Series: Series containing the data types of each column
    """
    return df.dtypes

def convert_column_type(df, column, dtype):
    """
    Convert a column to a specific data type.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the column
        column (str): Name of the column to convert
        dtype (type): Data type to convert to
        
    Returns:
        pandas.DataFrame: DataFrame with the column converted to the specified data type
    """
    try:
        df = df.copy()
        df[column] = df[column].astype(dtype)
        return df
    except Exception as e:
        print(f"Error converting column '{column}' to {dtype}: {e}")
        return df

def check_missing_values(df):
    """
    Check for missing values in a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame to check
        
    Returns:
        pandas.DataFrame: DataFrame containing the count and percentage of missing values for each column
    """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage': missing_percentage
    })
    return missing_df

def fill_missing_values(df, column, method='mean', value=None):
    """
    Fill missing values in a column.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the column
        column (str): Name of the column to fill
        method (str): Method to use for filling ('mean', 'median', 'mode', 'value')
        value: Value to use if method is 'value'
        
    Returns:
        pandas.DataFrame: DataFrame with missing values filled
    """
    df = df.copy()
    
    if method == 'mean':
        df[column] = df[column].fillna(df[column].mean())
    elif method == 'median':
        df[column] = df[column].fillna(df[column].median())
    elif method == 'mode':
        df[column] = df[column].fillna(df[column].mode()[0])
    elif method == 'value':
        df[column] = df[column].fillna(value)
    else:
        print(f"Unknown method: {method}")
    
    return df

def drop_missing_values(df, axis=0, how='any', thresh=None, subset=None):
    """
    Drop missing values from a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame to process
        axis (int): 0 for rows, 1 for columns
        how (str): 'any' or 'all'
        thresh (int): Require that many non-NA values
        subset (list): List of column names to consider
        
    Returns:
        pandas.DataFrame: DataFrame with missing values dropped
    """
    return df.dropna(axis=axis, how=how, thresh=thresh, subset=subset)

def fill_all_missing_values(df, method='mean'):
    """
    Fill all missing values in a DataFrame using the specified method.
    
    Args:
        df (pandas.DataFrame): DataFrame to process
        method (str): Method to use for filling ('mean', 'median', 'mode', 'ffill', 'bfill')
        
    Returns:
        pandas.DataFrame: DataFrame with all missing values filled
    """
    df = df.copy()
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if method == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif method == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif method == 'mode':
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # For categorical/string columns, use mode
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
            else:
                # For numeric columns
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
    elif method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    else:
        print(f"Unknown method: {method}")
    
    return df

def detect_outliers_iqr(df, column):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pandas.DataFrame): DataFrame to process
        column (str): Name of the column to check for outliers
        
    Returns:
        pandas.Series: Boolean series indicating outliers (True for outliers)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_outliers_iqr(df, column):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pandas.DataFrame): DataFrame to process
        column (str): Name of the column to remove outliers from
        
    Returns:
        pandas.DataFrame: DataFrame with outliers removed
    """
    outliers = detect_outliers_iqr(df, column)
    return df[~outliers]