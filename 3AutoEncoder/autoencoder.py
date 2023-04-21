from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
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

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding="same"))
        for _ in range(3):
            model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Flatten())

        model.add(Dense(49, activation='relu'))
        model.add(Reshape((7, 7, 1)))
        model.add(Conv2DTranspose(8, (3, 3), strides=1, activation='relu', padding='same'))
        for _ in range(2):
            model.add(Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'))

        model.add(Conv2DTranspose(1, (3, 3), activation='relu', padding='same'))

        model.compile(loss=keras.losses.mse,
                      optimizer=keras.optimizers.Adam(lr=.01),
                      metrics=['accuracy'])
        self.model = model
        print(model.summary())
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
            self.model.fit(x=x_train, y=x_train, batch_size=1024, epochs=epochs,
                           validation_data=(x_test, x_test), verbose=1,
                           callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True

    def generate_new(self, data: np.ndarray) -> np.ndarray:
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


if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    net = AutoEncoder(force_learn=True, file_name="models/autoencoder_small.h5")
    net.train(generator=gen, epochs=50)

    images_ds = gen.get_random_batch(training=False, batch_size=25000)[0][1:2, :, :, :]
    images = net.generate_new(images_ds)
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(images_ds[0, :, :, 0])
    axs[1].imshow(images[0, :, :, 0])
    plt.show()
    # I have no data generator (VAE or whatever) here, so just use a sampled set
    # print("Finished training")
    # img, labels = gen.get_random_batch(training=False,  batch_size=25000)
    # cov = net.check_class_coverage(data=img, tolerance=.98)
    # pred, acc = net.check_predictability(data=img, correct_labels=labels)
    # print(f"Coverage: {100*cov:.2f}%")
    # print(f"Predictability: {100*pred:.2f}%")
    # print(f"Accuracy: {100 * acc:.2f}%")