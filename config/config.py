from processing.data_management import load_dataset

dataset = '/dataset/data.csv'
data = load_dataset(dataset)
input_dim = data.shape[1]
latent_dim = 50 
num_synthetic_samples = 50000  
EPOCHS = 50
BATCH_SIZE = 32
TARGET = 'target'