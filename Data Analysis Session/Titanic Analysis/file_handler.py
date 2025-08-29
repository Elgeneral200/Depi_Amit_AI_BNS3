import sqlite3
import os
import pandas as pd

def read_csv(file_path, **kwargs):
    """
    Read data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pandas.read_csv
        
    Returns:
        pandas.DataFrame: The data from the CSV file
    """
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def read_excel(file_path, **kwargs):
    """
    Read data from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        **kwargs: Additional arguments to pass to pandas.read_excel
        
    Returns:
        pandas.DataFrame: The data from the Excel file
    """
    try:
        return pd.read_excel(file_path, **kwargs)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def read_json(file_path, **kwargs):
    """
    Read data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        **kwargs: Additional arguments to pass to pandas.read_json
        
    Returns:
        pandas.DataFrame: The data from the JSON file
    """
    try:
        return pd.read_json(file_path, **kwargs)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

def read_db(db_path, query, **kwargs):
    """
    Read data from a SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database
        query (str): SQL query to execute
        **kwargs: Additional arguments to pass to pandas.read_sql
        
    Returns:
        pandas.DataFrame: The data from the database
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(query, conn, **kwargs)
        conn.close()
        return df
    except Exception as e:
        print(f"Error reading database: {e}")
        return None

def read_file(file_path, **kwargs):
    """
    Read data from a file based on its extension.
    
    Args:
        file_path (str): Path to the file
        **kwargs: Additional arguments to pass to the appropriate read function
        
    Returns:
        pandas.DataFrame: The data from the file
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.csv':
        return read_csv(file_path, **kwargs)
    elif ext in ['.xlsx', '.xls']:
        return read_excel(file_path, **kwargs)
    elif ext == '.json':
        return read_json(file_path, **kwargs)
    elif ext == '.db':
        if 'query' not in kwargs:
            print("Error: SQL query must be provided for database files")
            return None
        query = kwargs.pop('query')
        return read_db(file_path, query, **kwargs)
    else:
        print(f"Unsupported file extension: {ext}")
        return None

def save_csv(df, file_path, **kwargs):
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        file_path (str): Path to save the CSV file
        **kwargs: Additional arguments to pass to pandas.to_csv
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        df.to_csv(file_path, **kwargs)
        return True
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return False

def save_excel(df, file_path, **kwargs):
    """
    Save a DataFrame to an Excel file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        file_path (str): Path to save the Excel file
        **kwargs: Additional arguments to pass to pandas.to_excel
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        df.to_excel(file_path, **kwargs)
        return True
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        return False

def save_json(df, file_path, **kwargs):
    """
    Save a DataFrame to a JSON file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        file_path (str): Path to save the JSON file
        **kwargs: Additional arguments to pass to pandas.to_json
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        df.to_json(file_path, **kwargs)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False

def save_db(df, db_path, table_name, if_exists='replace', **kwargs):
    """
    Save a DataFrame to a SQLite database.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        db_path (str): Path to the SQLite database
        table_name (str): Name of the table to save the data to
        if_exists (str): What to do if the table already exists ('fail', 'replace', or 'append')
        **kwargs: Additional arguments to pass to pandas.to_sql
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists=if_exists, **kwargs)
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False

def save_file(df, file_path, **kwargs):
    """
    Save a DataFrame to a file based on its extension.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        file_path (str): Path to save the file
        **kwargs: Additional arguments to pass to the appropriate save function
        
    Returns:
        bool: True if successful, False otherwise
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.csv':
        return save_csv(df, file_path, **kwargs)
    elif ext in ['.xlsx', '.xls']:
        return save_excel(df, file_path, **kwargs)
    elif ext == '.json':
        return save_json(df, file_path, **kwargs)
    elif ext == '.db':
        if 'table_name' not in kwargs:
            print("Error: table_name must be provided for database files")
            return False
        table_name = kwargs.pop('table_name')
        return save_db(df, file_path, table_name, **kwargs)
    else:
        print(f"Unsupported file extension: {ext}")
        return False