
import pandas as pd
def chk_types(df):
    '''Check data types and number of unique values for each column.'''
    dtypes = df.dtypes
    n_unique = df.nunique()
    return pd.DataFrame({'dtypes': dtypes, 'n_unique': n_unique}).transpose()