import matplotlib.pyplot as plt
import json
import numpy as np

with open('variational_autoencoder_training_data.json') as f:
   data = json.load(f)
training_kl = data["kl_loss"]
val_kl = data["val_kl_loss"]
val_im = data["val_image_reconstruction_loss"]
training_im = data["image_reconstruction_loss"]
validation = data['val_loss']
training = data['loss']
plt.plot(list(range(len(training))), np.log(training), label="Training loss")
plt.plot(list(range(len(training))), np.log(training_kl), label="Training KL loss")
plt.plot(list(range(len(training))), np.log(training_im), label="Training reconstruction loss")
plt.plot(list(range(len(validation))), np.log(validation), label="Validation loss")
plt.plot(list(range(len(validation))), np.log(val_kl), label="Validation KL loss")
plt.plot(list(range(len(validation))), np.log(val_im), label="Validation reconstruction loss")
plt.legend()
plt.title('Variational autoencoder with 8')
plt.savefig("vae_training_plot_.png")