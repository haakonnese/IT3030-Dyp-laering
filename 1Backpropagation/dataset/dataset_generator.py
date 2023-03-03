import math
import random

import dataset.doodler_forall


class DatasetGenerator:
    @staticmethod
    def generate_dataset(n: int,
                         number: int,
                         noise: float,
                         wr: list[float],
                         hr: list[float],
                         random_seed: int = None,
                         center: bool = True,
                         show_n_random_images: int = 0,
                         flat: bool = False,
                         train: float = 0.7,
                         val: float = 0.2,
                         test: float = 0.1
                         ):
        """
        Generates training, validation and test datasets with 4 different shapes using the Doodler class
        :param n: size of grid. Must be in range 10 <= n <= 50
        :param number: of images to generate
        :param noise: a number between 0 and 1, specifying the probability that a single bit is flipped
        :param wr: a list of two numbers, lower and upper bound of the portion of the width of the image to the grid size
        :param hr: a list of two numbers, lower and upper bound of the portion of the height of the image to the grid size
        :param random_seed: if set, the dataset generated will be reproducible
        :param center: if the shape shall be centered in the grid
        :param show_n_random_images: number of random images to show
        :param flat: representing the image as a flat list or a 2d array
        :param train: partition of training data
        :param val: partition of validation data
        :param test: partition of test data
        :return: a tuple with training set, validation set and test set
        """
        if train + val + test - 1 > 1e-16:
            raise ValueError("Som of `train`, `val` and `test` must be 1")
        if n not in range(10, 51):
            raise ValueError("'n' must be an int in the range 10 <= n <= 50")

        if noise < 0 or noise > 1:
            raise ValueError("'noise' must be a number between 0 and 1")
        if number < 0 or type(number) != int:
            raise ValueError("'number' must be a positive integer")
        if len(wr) != 2 or len(hr) != 2:
            raise ValueError("'hr' and 'wr' must have a length of exactly 2")

        images, y, labels, dims, flat = dataset.doodler_forall.gen_standard_cases(count=number,
                                                                                  rows=n,
                                                                                  cols=n,
                                                                                  types=['ball', 'frame', 'flower',
                                                                                         'spiral'],
                                                                                  wr=wr,
                                                                                  hr=hr,
                                                                                  noise=noise,
                                                                                  cent=center,
                                                                                  show=False,
                                                                                  flat=flat,
                                                                                  random_seed=random_seed)
        random.seed(random_seed)
        random_images = random.sample(range(0, number), show_n_random_images)
        images_to_show = images[random_images, :]
        labels_to_show = [labels[random_image] for random_image in random_images]
        if show_n_random_images > 0:
            dataset.doodler_forall.show_doodle_cases((images_to_show, y, labels_to_show, dims, flat))
        train_number = math.floor(number * train)
        val_number = math.floor(number * val)
        test_number = number - train_number - val_number
        train_data = images[0:train_number, :], y[0:train_number, :]
        val_data = images[train_number:train_number + val_number, :], y[train_number:train_number + val_number, :]
        test_data = images[-test_number - 1: - 1, :], y[-test_number - 1: - 1, :]
        return train_data, val_data, test_data
