from tensorflow import keras
import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm, poisson


class DataGenerator_tf(keras.utils.Sequence):
    """Generates data for Keras  with tensorflow native operations."""

    def __init__(self, X, y, transform_X_fun=None, transform_y_fun=None, batch_size=32, shuffle=True):
        """Initialization.

        :param X: Values of independent variable eg. pictures.
        :param y: Labels.
        :param transform_X_fun: Function applied to each entry in X_batch before returning the batch.
            For example, it can be noise.
        :param transform_y_fun: Function applied to each entry in y_batch before returning the batch.
        :param batch_size: Batch size.
        :param shuffle: Whether to shuffle the original order of entries in X and y.
        """
        self.X = tf.convert_to_tensor(X)
        self.n_samples = X.shape[0]
        self.input_dim = X.shape[1:]
        self.y = y
        self.output_dim = y.shape[-1]
        self.transform_X_fun = transform_X_fun
        self.transform_y_fun = transform_y_fun
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.n_samples / self.batch_size))-1

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X_batch, y_batch = self.__data_generation(indexes)

        return X_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = tf.range(self.n_samples)

        if self.shuffle:
            tf.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples with noise etc."""  # X_batch : (batch_size, *input_dim)

        if self.transform_X_fun is not None:
            X_batch = self.transform_X_fun(tf.gather(self.X, indices=indexes))
        else:
            X_batch = tf.gather(self.X, indices=indexes)

        if self.transform_y_fun is not None:
            y_batch = self.transform_y_fun(tf.gather(self.y, indices=indexes))
        else:
            y_batch = tf.gather(self.y, indices=indexes)

        return X_batch, y_batch


def random_contrast_tf(X, contrast_min=0.2, contrast_max=2.0):
    X = (X + 1.) * 127.5
    X = tf.image.random_contrast(X, 0.2, 2)
    X = X / 127.5 - 1.
    return X


def random_flip_lrud_tf(X):
    X = tf.image.random_flip_left_right(X)
    X = tf.image.random_flip_up_down(X)
    return X


def random_brightness_tf(X, brightness_max_delta=0.2):
    X = (X + 1.) * 127.5
    X = tf.image.random_brightness(X, brightness_max_delta)
    X = X / 127.5 - 1.
    return X


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10., dtype=numerator.dtype))
    return numerator / denominator


def random_background_noise_for_preprocess_1_tf(X, threshold, noise_level=1e-6):
    """
    Adds additive background noise to the data fed through preprocess_1 (already in the log space and range [-1, 1]).
    :param X: pictures in range(-1, 1)
    :param threshold: threshold used in preprocess_2.
    :param noise_level: additive noise level in units of original, normalized signal.
    :return: noised X
    """
    X = tf.math.pow(10., -tf_log10(threshold) * X)
    X += tf.random.truncated_normal(mean=noise_level, stddev=noise_level/2, shape=X.shape)

    X = tf_log10(X)
    X = X / (-tf_log10(threshold))

    return X


def random_background_noise_for_preprocess_2_tf(X, threshold, noise_level=1e-4):
    """
    Adds additive background noise to the data fed through preprocess_2 (already in the log space and range [-1, 1]).
    :param X: pictures in range(-1, 1)
    :param threshold: threshold used in preprocess_2.
    :param noise_level: additive noise level in units of the maximal signal.
    :return: noised X
    """
    threshold = tf.constant(threshold, dtype=X.dtype)
    X = tf.math.pow(tf.constant(10., dtype=X.dtype),
                    tf.constant(0.5, dtype=X.dtype) * tf_log10(threshold) * (
                            tf.constant(1., shape=X.shape, dtype=X.dtype) - X))
    X += tf.random.truncated_normal(mean=noise_level, stddev=noise_level/2., shape=X.shape, dtype=X.dtype)

    X = tf_log10(X)
    X = X + (-tf_log10(threshold) / tf.constant(2., dtype=X.dtype))
    X = X / (-tf_log10(threshold) / tf.constant(2., dtype=X.dtype))

    return X


def random_poisson_noise_for_preprocess_2_tf(X, threshold, max_counts_per_pixel=1e4):
    """
    Adds Poisson noise to the data fed through preprocess_2 (already in the log space and range [-1, 1]).
    :param X: pictures in range(-1, 1)
    :param threshold: threshold used in preprocess_2.
    :param max_counts_per_pixel: maximal count of photoelectrons per pixel.
    :return: noisy X
    """
    threshold = tf.constant(threshold, dtype=X.dtype)
    mcpp = tf.constant(max_counts_per_pixel, dtype=X.dtype)
    X = tf.math.pow(tf.constant(10., dtype=X.dtype),
                    tf.constant(0.5, dtype=X.dtype) * tf_log10(threshold) * (
                            tf.constant(1., shape=X.shape, dtype=X.dtype) - X))
    X = (tf.random.poisson([1], X*mcpp, dtype=X.dtype)[0] + tf.constant(1., dtype=X.dtype)) / mcpp
    X = tf_log10(X)
    X = X + (-tf_log10(threshold) / tf.constant(2., dtype=X.dtype))
    X = X / (-tf_log10(threshold) / tf.constant(2., dtype=X.dtype))

    return X


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras model with numpy array operations."""

    def __init__(self, X, y, transform_X_fun=None, transform_y_fun=None, batch_size=32, shuffle=True):
        """Initialization.

        :param X: Values of independent variable eg. pictures.
        :param y: Labels.
        :param transform_X_fun: Function applied to each entry in X_batch before returning the batch.
            For example, it can be noise.
        :param transform_y_fun: Function applied to each entry in y_batch before returning the batch.
        :param batch_size: Batch size.
        :param shuffle: Whether to shuffle the original order of entries in X and y.
        """
        self.X = X
        self.n_samples = X.shape[0]
        self.input_dim = X.shape[1:]
        self.y = y
        self.output_dim = y.shape[-1]
        self.transform_X_fun = transform_X_fun
        self.transform_y_fun = transform_y_fun
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.n_samples / self.batch_size))-1

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X_batch, y_batch = self.__data_generation(indexes)

        return X_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.n_samples)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples with noise etc."""  # X_batch : (batch_size, *input_dim)

        if self.transform_X_fun is not None:
            X_batch = self.transform_X_fun(self.X[indexes])
        else:
            X_batch = self.X[indexes]

        if self.transform_y_fun is not None:
            y_batch = self.transform_y_fun(self.y[indexes])
        else:
            y_batch = self.y[indexes]

        return X_batch, y_batch


def random_background_noise_for_preprocess_1(X, threshold, noise_level=1e-6):
    """
    Adds additive background noise to the data fed through preprocess_1 (already in the log space and range [-1, 1]).
    :param X: pictures in range(-1, 1)
    :param threshold: threshold used in preprocess_2.
    :param noise_level: additive noise level in units of original, normalized signal.
    :return: noised X
    """
    X = np.power(10., -np.log10(threshold) * X)
    X += np.random.truncated_normal(mean=noise_level, stddev=noise_level/2, shape=X.shape)

    X = np.log10(X)
    X = X / (-np.log10(threshold))

    return X


def random_background_noise_for_preprocess_2(X, threshold, noise_level=1e-4):
    """
    Adds additive background noise to the data fed through preprocess_2 (already in the log space and range [-1, 1]).
    :param X: pictures in range(-1, 1)
    :param threshold: threshold used in preprocess_2.
    :param noise_level: additive noise level in units of the maximal signal.
    :return: noised X
    """
    X = np.power(10., 0.5 * np.log10(threshold) * (1. - X))
    X += truncnorm.rvs(0, 1, loc=noise_level, scale=noise_level, size=X.shape)

    X = np.log10(X)
    X = X + (-np.log10(threshold) / 2.)
    X = X / (-np.log10(threshold) / 2.)

    return X


