from stacked_mnist import StackedMNISTData, DataMode
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Conv2D, Conv2DTranspose, Reshape, Activation, Dropout
import numpy as np
import json
import matplotlib.pyplot as plt
from verification_net import VerificationNet

class AutoEncoder:
    def __init__(self, force_learn: bool = False, file_name: str = "models/autoencoder.h5", latent_dim=2) -> None:
        """
        Define model and set some parameters.
        The model is  made for classifying one channel only -- if we are looking at a
        more-channel image we will simply do the thing one-channel-at-the-time.
        """
        self.force_relearn = force_learn
        self.file_name = file_name
        self.latent_dim = latent_dim
        input_img = keras.Input(shape=(28, 28, 1))

        encoder = Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(input_img)
        encoder = BatchNormalization()(encoder)
        # encoder = Activation(relu)(encoder)
        encoder = Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(encoder)
        encoder = BatchNormalization()(encoder)
        # encoder = Activation(relu)(encoder)
        for _ in range(8):
            encoder = Conv2D(128, (3, 3), strides=1, activation="relu", padding="same")(encoder)
            encoder = BatchNormalization()(encoder)
            # encoder = Activation(relu)(encoder)

        encoder = Flatten()(encoder)
        encoder = Dense(self.latent_dim, activation="sigmoid")(encoder)

        encoder_model = Model(input_img, encoder)

        input_decoder = keras.Input(shape=(self.latent_dim,))
        decoder = Dense(49, activation='relu')(input_decoder)
        decoder = Reshape((7, 7, 1))(decoder)
        for _ in range(8):
            decoder = Conv2DTranspose(128, (3, 3), activation="relu", strides=1, padding='same')(decoder)
            decoder = BatchNormalization()(decoder)
            # encoder = Activation(relu)(encoder)

        for channels in [128, 64]:
            decoder = Conv2DTranspose(channels, (3, 3), activation="relu", strides=2, padding='same')(decoder)
            decoder = BatchNormalization()(decoder)
            # encoder = Activation(relu)(encoder)

        decoder = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(decoder)
        decoder_model = Model(input_decoder, decoder)

        autoencoder_input = keras.Input(shape=(28, 28, 1))
        encoded = encoder_model(autoencoder_input)
        decoded = decoder_model(encoded)
        autoencoder_model = Model(inputs=autoencoder_input, outputs=decoded)

        autoencoder_model.compile(loss=keras.losses.binary_crossentropy,
                                  optimizer=keras.optimizers.Adam(lr=1e-3),
                                  metrics=['accuracy'])
        self.model = autoencoder_model
        self.decoder = decoder_model
        print(encoder_model.summary())
        print(decoder_model.summary())
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
            history = self.model.fit(x=x_train, y=x_train, batch_size=2048, epochs=epochs,
                                     validation_data=(x_test, x_test), verbose=1, shuffle=True)
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
            random_input = np.random.uniform(0, 1, size=(number_of_images, self.latent_dim))
            random_input = np.tanh(random_input)
            channel_prediction = self.decoder.predict(random_input)
            generated_images[:, :, :, channel] = channel_prediction[:, :, :, 0]
        return generated_images

    def anomaly_detection(self, data: np.ndarray):
        no_channels = data.shape[-1]
        print(data.shape)
        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError("Model is not trained, so makes no sense to try to use it")

        generated_images = np.zeros(data.shape)
        for channel in range(no_channels):
            channel_prediction = self.model.predict(data[:, :, :, [channel]])
            generated_images[:, :, :, channel] = channel_prediction[:, :, :, 0]
        # generate error for each image, shape is batch, 28, 28, no_channels. The error should be cross entropy
        # between the original image and the generated image. It should be one number for each image
        epsilon = 1e-7  # to avoid division by zero errors
        generated_images = np.clip(generated_images, epsilon, 1.0 - epsilon)  # clip values to avoid NaNs in log
        return generated_images, keras.backend.mean(keras.losses.binary_crossentropy(data, generated_images), axis=[1,2])


if __name__ == "__main__":
    mode = DataMode.COLOR_BINARY_MISSING
    if mode == DataMode.MONO_BINARY_COMPLETE or mode == DataMode.MONO_BINARY_MISSING:
        tolerance = 0.8
        mono_color = "mono"
        channels_view = 0

    else:
        tolerance = 0.5
        mono_color = "color"
        channels_view = slice(0,3)
    if mode == DataMode.MONO_BINARY_COMPLETE or mode == DataMode.COLOR_BINARY_COMPLETE:
        filename = "models/autoencoder.h5"
        png_extra = "complete_"
        printing = "with 8"
    else:
        filename = "models/autoencoder_anomalies.h5"
        png_extra = "anomalies_"
        printing = "without 8"


    gen = StackedMNISTData(mode=mode, default_batch_size=2048)
    net = AutoEncoder(force_learn=False, file_name=filename, latent_dim=2)
    net.train(generator=gen, epochs=300)
    print("Finished training")
    verification_net = VerificationNet(force_learn=False, file_name="models/verification_model.h5")

    show_number_of_images = 10
    number_of_anomalies = 100
    number_generate_images = 10000

    images_ds, classes = gen.get_random_batch(training=False, batch_size=25000)
    images_ds = images_ds.astype(float)
    images = net.reconstruct(images_ds)
    _, axs = plt.subplots(2, show_number_of_images)
    images = images.astype(float)

    for i in range(show_number_of_images):
        axs[0][i].set_xticks([])
        axs[1][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[1][i].set_yticks([])
        axs[0][i].imshow(images_ds[i, :, :, channels_view])
        axs[1][i].imshow(images[i, :, :, channels_view])
    plt.savefig(f"autoencoder_{mono_color}_{png_extra}test_data.png")

    cov = verification_net.check_class_coverage(data=images, tolerance=tolerance)
    pred, acc = verification_net.check_predictability(data=images, correct_labels=classes, tolerance=tolerance)
    print(f"Predicting test-cases for data trained {printing}. For the {mono_color}-case")
    print(f"Coverage: {100 * cov:.2f}%")
    print(f"Predictability: {100 * pred:.2f}%")
    print(f"Accuracy: {100 * acc:.2f}%")

    _, axs = plt.subplots(1, show_number_of_images)
    images = net.generate_random_images(number_generate_images, no_channels=gen.channels)
    images = images.astype(float)
    for i in range(show_number_of_images):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].imshow(images[i, :, :, channels_view])
    plt.savefig(f"autoencoder_{mono_color}_{png_extra}generator.png")
    cov_gen = verification_net.check_class_coverage(data=images, tolerance=tolerance)
    pred_gen, _ = verification_net.check_predictability(data=images, tolerance=tolerance)
    print(f"Coverage generator: {100 * cov_gen:.2f}%")
    print(f"Predictability generator: {100 * pred_gen:.2f}%")


    anomaly_images, cross_entropy_error = net.anomaly_detection(images_ds)
    most_error = np.argsort(cross_entropy_error, axis=None)[::-1][0:number_of_anomalies]
    _, axs = plt.subplots(2, show_number_of_images)

    for i in range(show_number_of_images):
        axs[0][i].set_xticks([])
        axs[1][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[1][i].set_yticks([])
        axs[0][i].imshow(images_ds[most_error[i], :, :, channels_view])
        axs[1][i].imshow(anomaly_images[most_error[i], :, :, channels_view])
    plt.savefig(f"autoencoder_{mono_color}_{png_extra}anomalies.png")
    class_errors = []
    for i, error_img in enumerate(most_error):
        class_errors.append(classes[error_img])

    print("Printing classes with anomalies")
    for i in range(10 if mono_color == "mono" else 1000):
        if class_errors.count(i) > 0:
            print(f"Anomalies: Number of class {i:03d}: {class_errors.count(i)}")