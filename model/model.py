import keras
import tensorflow as tf
from keras import backend as K
tf.compat.v1.disable_eager_execution()
from config.config import latent_dim, input_dim

# Reparameterization trick
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

def vae_model():
    # Encoder network
    encoder_inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(encoder_inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)  # Output mean
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)  # Output log variance

    z = Sampling()([z_mean, z_log_var])

    # Decoder network
    decoder_inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(32, activation='relu')(decoder_inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    decoder_outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)  # Use 'linear' activation for better diversity

    # Define the VAE models
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name='decoder')

    # VAE model
    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = tf.keras.Model(encoder_inputs, vae_outputs, name='vae')

    # Define the VAE loss function (VAE-specific loss)
    reconstruction_loss = tf.keras.losses.mean_squared_error(encoder_inputs, vae_outputs)
    kl_loss = -0.5 * tf.keras.backend.mean(1 + z_log_var - tf.keras.backend.exp(z_log_var) - tf.keras.backend.square(z_mean), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return vae, decoder