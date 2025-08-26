import pandas as pd
def read_file(path):
    df = pd.read_csv(path)
    return df.head()