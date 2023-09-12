# VAE for Data Generation
Using Variational Auto Encoders for synthetic data generation.
## Introduction
An autoencoder is a neural network that learns to copy its input to its output. It consists of an encoder that compresses the input into a latent space representation and a decoder that reconstructs the input from the latent space.
Variational autoencoders (VAEs) are a type of autoencoder that focus on dimensionality reduction. Unlike traditional autoencoders, VAEs represent the latent space as a distribution rather than a single point. This is achieved by incorporating a regularization term and a reconstruction error.

![image](https://github.com/parakatta/vae-data-generation/assets/83866928/5649832b-2bbd-4c1d-ab8d-f289a2c5cdc5)  

The decoder network in VAEs can generate new data by sampling from the prior distribution in the latent space. VAEs offer a powerful tool for generative modeling and data reconstruction.

 ![image](https://github.com/parakatta/vae-data-generation/assets/83866928/af61ff3f-d218-47a4-85ae-7e7f842702e6)
## Application
While traditional autoencoders find applications in image denoising, dimensionality reduction, and data compression, VAEs excel in image and time series data generation. They are particularly useful for anomaly detection, as the latent attributes cannot accurately reconstruct images that are not part of the training dataset, enabling the identification of anomalies.

## Implementation  
Install dependencies.
```
pip install -r requirements.txt  
```
Then make necessary changes in the config/config.py file for custom dataset or parameters.  
Run main.py

```
python main.py
```
The similarity scores for the dataset will be returned.
