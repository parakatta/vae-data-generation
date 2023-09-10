import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
tf.compat.v1.disable_eager_execution()
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from config.config import TARGET

def rmse(y1,y2):
    return(np.sqrt(metrics.mean_squared_error(y1,y2)))

def score_new(generated_data,data_new):
    X_train = generated_data.drop(columns=[TARGET])  
    y_train = generated_data[TARGET] 
    X_train_pure=data_new.drop(columns=[TARGET])  
    y_train_pure=data_new[TARGET] 
    model = RandomForestRegressor(n_estimators=25, max_depth=7, n_jobs=-1, random_state=42) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train_pure)
    rmse_pure = rmse(y_train_pure, y_pred)
    return rmse_pure


def check_similarity(data, generated_data):
    similarity_matrix = cosine_similarity(data, generated_data)
    similarity_df = pd.DataFrame(similarity_matrix, columns=generated_data.index)
    sorted_dataset2 = generated_data.iloc[similarity_df.mean(axis=1).sort_values(ascending=False).index]
    selected_rows = sorted_dataset2.head(3500-data.shape[0])
    return selected_rows


def scale_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    min_values = scaler.data_min_  # Minimum values from the scaler
    max_values = scaler.data_max_ 
    return min_values, max_values