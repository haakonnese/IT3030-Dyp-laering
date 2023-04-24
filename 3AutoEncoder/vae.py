from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Conv2D, Activation, Conv2DTranspose, Reshape
import numpy as np
from keras.callbacks import TensorBoard
import os
import json
from verification_net import VerificationNet

import matplotlib.pyplot as plt


class VariationalAutoEncoder:
    def __init__(self, force_learn: bool = False, file_name: str = "models/variational_autoencoder.h5",
                 latent_dim=50, from_start: bool = False) -> None:
        """
        Define model and set some parameters.
        The model is  made for classifying one channel only -- if we are looking at a
        more-channel image we will simply do the thing one-channel-at-the-time.
        """
        self.force_relearn = force_learn
        self.file_name = file_name
        self.latent_dim = latent_dim
        input_img = keras.Input(shape=(28, 28, 1))
        encoder = Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(input_img)
        encoder = BatchNormalization()(encoder)
        encoder = Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(encoder)
        encoder = BatchNormalization()(encoder)

        for _ in range(8):
            encoder = Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(encoder)
            encoder = BatchNormalization()(encoder)

        encoder = Flatten()(encoder)

        encoder_mean = Dense(latent_dim)(encoder)
        encoder_log_var = Dense(latent_dim)(encoder)

        encoder = keras.layers.Lambda(self.z_layer)([encoder_mean, encoder_log_var])

        encoder_model = Model(input_img, [encoder_mean, encoder_log_var, encoder])

        input_decoder = keras.Input(shape=(latent_dim,))
        decoder = Dense(49, activation='relu')(input_decoder)
        decoder = Reshape((7, 7, 1))(decoder)
        for _ in range(8):
            decoder = Conv2DTranspose(64, (3, 3), activation="relu", strides=1, padding='same')(decoder)
            decoder = BatchNormalization()(decoder)

        for channels in [64, 32]:
            decoder = Conv2DTranspose(channels, (3, 3), activation="relu", strides=2, padding='same')(decoder)
            decoder = BatchNormalization()(decoder)

        decoder = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(decoder)

        decoder_model = Model(input_decoder, decoder)

        encoded = encoder_model(input_img)[2]
        decoded = decoder_model(encoded)
        vae_model = Model(inputs=input_img, outputs=decoded)

        # Calculate the difference between the images. The same as for autoencoder.
        image_difference_loss = keras.losses.binary_crossentropy(input_img, decoded)
        image_difference_loss = keras.backend.mean(keras.backend.sum(image_difference_loss, axis=[1, 2]))

        # add the kl_divergence loss term. Defined as https://arxiv.org/abs/1312.6114
        kl_loss = 1 + encoder_log_var - keras.backend.square(encoder_mean) - keras.backend.exp(encoder_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_model.add_loss(keras.backend.mean(image_difference_loss + kl_loss))
        vae_model.add_metric(value=kl_loss, name="kl_loss")
        vae_model.add_metric(value=image_difference_loss, name="image_reconstruction_loss")
        vae_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))
        self.model = vae_model
        self.decoder = decoder_model
        print(encoder_model.summary())
        print(decoder_model.summary())
        print(vae_model.summary())
        self.from_start = from_start
        if not self.from_start:
            self.done_training = self.load_weights()
        else:
            self.done_training = False

    def z_layer(self, args):
        z_mean, z_log_var = args

        # N(z_mean, z_sd) = N(z_mean + sqrt(z_var)) = z_mean + exp(z_log_var * 0.5) * N(0,1)
        # Just sample one number from the normal distribution and shift it
        epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], self.latent_dim))
        return z_mean + keras.backend.exp(z_log_var * 0.5) * epsilon

    def load_weights(self):
        # noinspection PyBroadException
        try:
            self.model.load_weights(filepath=self.file_name)
            # print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(f"Could not read weights for autoencoder from file. Must retrain...")
            done_training = False

        return done_training

    def train(self, generator: StackedMNISTData, epochs: np.int = 10) -> bool:
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """
        if not self.from_start:
            self.done_training = self.load_weights()
        else:
            self.done_training = False

        if self.force_relearn or self.done_training is False:
            # Get hold of data
            x_train, _ = generator.get_full_data_set(training=True)
            x_test, _ = generator.get_full_data_set(training=False)

            # "Translate": Only look at "red" channel; only use the last digit. Use one-hot for labels during training
            x_train = x_train[:, :, :, [0]]
            x_test = x_test[:, :, :, [0]]
            # Fit model
            print(self.file_name.split(".")[0].split("/")[1])
            history = self.model.fit(x=x_train, y=x_train, batch_size=1024, epochs=epochs,
                                     validation_data=(x_test, x_test), verbose=1)
            self.model.save_weights(filepath=self.file_name)
            with open(f'{self.file_name.split(".")[0].split("/")[1]}_training_data.json', 'w') as f:
                json.dump(history.history, f, ensure_ascii=False)
            self.done_training = True

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """
        Predict the classes of some specific data-set. This is basically prediction using keras, but
        this method is supporting multi-channel inputs.
        Since the model is defined for one-channel inputs, we will here do one channel at the time.

        The rule here is that channel 0 define the "ones", channel 1 defines the tens, and channel 2
        defines the hundreds.

        Since we later need to know what the "strength of conviction" for each class-assessment we will
        return both classifications and the belief of the class.
        For multi-channel images, the belief is simply defined as the probability of the allocated class
        for each channel, multiplied.
        """
        no_channels = data.shape[-1]
        print(data.shape)
        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError("Model is not trained, so makes no sense to try to use it")

        generated_images = np.zeros(data.shape)
        for channel in range(no_channels):
            channel_prediction = self.model.predict(data[:, :, :, [channel]])
            generated_images[:, :, :, channel] = channel_prediction[:, :, :, 0]
        return generated_images

    def generate_random_images(self, number_of_images: int = 10, no_channels=1) -> np.ndarray:
        """
        Generate random images using the decoder part of the autoencoder.
        """
        generated_images = np.zeros((number_of_images, 28, 28, no_channels))
        for channel in range(no_channels):
            random_input = np.random.normal(size=(number_of_images, self.latent_dim))
            channel_prediction = self.decoder.predict(random_input)
            generated_images[:, :, :, channel] = channel_prediction[:, :, :, 0]
        return generated_images


if __name__ == "__main__":
    mode = DataMode.MONO_BINARY_COMPLETE
    if mode == DataMode.MONO_BINARY_COMPLETE or mode == DataMode.MONO_BINARY_MISSING:
        tolerance = 0.8
    else:
        tolerance = 0.5
    if mode == DataMode.MONO_BINARY_COMPLETE or mode == DataMode.COLOR_BINARY_COMPLETE:
        filename = "models/variational_autoencoder.h5"
    else:
        filename = "models/variational_autoencoder_anomalies.h5"
    gen = StackedMNISTData(mode=mode, default_batch_size=2048)
    net = VariationalAutoEncoder(force_learn=True, from_start=True, file_name=filename, latent_dim=2)
    net.train(generator=gen, epochs=1000)

    verification_net = VerificationNet(force_learn=False, file_name="models/verification_model.h5")
    show_number_of_images = 10
    images_ds, classes = gen.get_random_batch(training=False, batch_size=25000)
    images = net.reconstruct(images_ds)
    _, axs = plt.subplots(2, show_number_of_images)
    images_ds = images_ds.astype(float)
    images = images.astype(float)

    cov = verification_net.check_class_coverage(data=images, tolerance=tolerance)
    pred, acc = verification_net.check_predictability(data=images, correct_labels=classes, tolerance=tolerance)
    print(f"Coverage: {100 * cov:.2f}%")
    print(f"Predictability: {100 * pred:.2f}%")
    print(f"Accuracy: {100 * acc:.2f}%")

    for i in range(show_number_of_images):
        axs[0][i].set_xticks([])
        axs[1][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[1][i].set_yticks([])
        axs[0][i].imshow(images_ds[i, :, :, 0])
        axs[1][i].imshow(images[i, :, :, 0])
    plt.savefig("vae_test_data.png")
    # plt.show()

    _, axs = plt.subplots(1, show_number_of_images)
    images = net.generate_random_images(show_number_of_images, no_channels=gen.channels)
    images = images.astype(float)
    print(images.shape)
    for i in range(show_number_of_images):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].imshow(images[i, :, :, 0])
    plt.savefig("vae_generator.png")
