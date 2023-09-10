import pandas as pd

def load_dataset(path):
    data = pd.read_csv(path)
    return data