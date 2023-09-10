 # Adjust the latent dimension
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
#from keras.layers import BatchNormalization
import numpy as np
tf.compat.v1.disable_eager_execution()
import processing.metrics as mm
from config.config import *
from model.model import vae_model
from processing.data_management import load_dataset
# Train the VAE on 'data_new'

data = load_dataset(dataset)
model, decoder = vae_model()
model.fit(data, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
synthetic_noise = np.random.normal(0, 1, size=(num_synthetic_samples, latent_dim))
generated_data = decoder.predict(synthetic_noise)
min_values,max_values = mm.scale_data(data)
generated_data = np.clip(generated_data, min_values, max_values)
generated_data=pd.DataFrame(generated_data,columns=data.columns)
print("ans",mm.score_new(generated_data,data))
selected_rows = mm.check_similarity(data, generated_data)
print("end",mm.score_new(selected_rows,data))