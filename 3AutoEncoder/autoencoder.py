from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape
import numpy as np
from keras.callbacks import TensorBoard
import os
import json

PLOT_IMAGES = True
if PLOT_IMAGES:
    import matplotlib.pyplot as plt


class AutoEncoder:
    def __init__(self, force_learn: bool = False, file_name: str = "models/autoencoder.h5") -> None:
        """
        Define model and set some parameters.
        The model is  made for classifying one channel only -- if we are looking at a
        more-channel image we will simply do the thing one-channel-at-the-time.
        """
        self.force_relearn = force_learn
        self.file_name = file_name
        input_img = keras.Input(shape=(28, 28, 1))
        encoder = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding="same")(input_img)
        for _ in range(3):
            encoder = Conv2D(32, (3, 3), activation='relu', padding="same")(encoder)
            encoder = MaxPooling2D(pool_size=(2, 2))(encoder)

        encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
        encoder = MaxPooling2D((2, 2), padding='same')(encoder)
        encoder = keras.layers.Activation(keras.activations.tanh)(encoder)
        encoder = Flatten()(encoder)

        encoder_model = Model(input_img, encoder)

        input_decoder = keras.Input(shape=(32,))
        decoder = Dense(49, activation='relu')(input_decoder)
        decoder = Reshape((7, 7, 1))(decoder)
        decoder = Conv2DTranspose(8, (3, 3), strides=1, activation='relu', padding='same')(decoder)
        for _ in range(2):
            decoder = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(decoder)

        decoder = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(decoder)
        decoder_model = Model(input_decoder, decoder)

        autoencoder_input = keras.Input(shape=(28, 28, 1))
        encoded = encoder_model(autoencoder_input)
        decoded = decoder_model(encoded)
        autoencoder_model = Model(inputs=autoencoder_input, outputs=decoded)

        autoencoder_model.compile(loss=keras.losses.mse,
                                  optimizer=keras.optimizers.Adam(lr=.01),
                                  metrics=['accuracy'])
        self.model = autoencoder_model
        self.decoder = decoder_model
        print(autoencoder_model.summary())
        self.done_training = self.load_weights()

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
        self.done_training = self.load_weights()

        if self.force_relearn or self.done_training is False:
            # Get hold of data
            x_train, _ = generator.get_full_data_set(training=True)
            x_test, _ = generator.get_full_data_set(training=False)

            # "Translate": Only look at "red" channel; only use the last digit. Use one-hot for labels during training
            x_train = x_train[:, :, :, [0]]
            x_test = x_test[:, :, :, [0]]
            # Fit model
            history = self.model.fit(x=x_train, y=x_train, batch_size=1024, epochs=epochs,
                                     validation_data=(x_test, x_test), verbose=1)
            self.model.save_weights(filepath=self.file_name)
            import json
            with open('training.json', 'w') as f:
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
            random_input = np.random.uniform(-1, 1, size=(number_of_images, 32))
            channel_prediction = self.decoder.predict(random_input)
            generated_images[:, :, :, channel] = channel_prediction[:, :, :, 0]
        return generated_images


if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
    net = AutoEncoder(force_learn=False, file_name="models/autoencoder_small.h5")
    net.train(generator=gen, epochs=1)
    show_number_of_images = 10
    if PLOT_IMAGES:
        images_ds = gen.get_random_batch(training=False, batch_size=25000)[0][0:show_number_of_images, :, :, :]
        images = net.reconstruct(images_ds)
        _, axs = plt.subplots(2, show_number_of_images)
        images_ds = images_ds.astype(float)
        images = images.astype(float)
        print(images[0, :, :, 0])

        for i in range(show_number_of_images):
            axs[0][i].set_xticks([])
            axs[1][i].set_xticks([])
            axs[0][i].set_yticks([])
            axs[1][i].set_yticks([])
            axs[0][i].imshow(images_ds[i, :, :, :])
            axs[1][i].imshow(images[i, :, :, :])
        plt.show()

        _, axs = plt.subplots(1, show_number_of_images)
        images = net.generate_random_images(show_number_of_images, no_channels=gen.channels)
        images = images.astype(float)
        print(images[0, :, :, 0])
        for i in range(show_number_of_images):
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].imshow(images[i, :, :, :])
        plt.show()
    # I have no data generator (VAE or whatever) here, so just use a sampled set
    # print("Finished training")
    # img, labels = gen.get_random_batch(training=False,  batch_size=25000)
    # cov = net.check_class_coverage(data=img, tolerance=.98)
    # pred, acc = net.check_predictability(data=img, correct_labels=labels)
    # print(f"Coverage: {100*cov:.2f}%")
    # print(f"Predictability: {100*pred:.2f}%")
    # print(f"Accuracy: {100 * acc:.2f}%")
