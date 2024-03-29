import numpy as np
from skimage.transform import resize


def preprocess_1(PDFs, normalize_X_fun=lambda x: x, threshold=1e-10, downsample_1=1, downsample_2=1):
    """
    Preprocess PDFs:
        1. Take every ``downsample_1`` entry across axis 1 and every ``downsample_2`` entry across axis 2.
        2. Normalize using ``normalize_X_fun`` function.
        3. Everything below ``threshold`` is set to ``threshold``.
        4. X = np.log10(X)
        5. X = X / (-np.log10(threshold))
        6. Resize to shape (PDFs.shape[0], 299, 299, 3) where channels are copied from one channel initially.

    :param PDFs: Pictures with probability distributions (in the linear scale).
    :param normalize_X_fun: Function to normalize the PDF.
    :param threshold: Lower bound for the normalized values.
    :param downsample_1: Take every ``downsample_1`` entry across axis 1.
    :param downsample_2: Take every ``downsample_2`` entry across axis 2.
    :return: function that acts on PDFs and returns shape (PDFs.shape[0], 299, 299, 3).
    """
    X = PDFs[:, ::downsample_1, ::downsample_2]

    X = normalize_X_fun(X)
    X[np.where(X < threshold)] = threshold
    X = np.log10(X)
    X = X / (-np.log10(threshold))
    X[np.where(X > 1.)] = 1
    X = np.array([resize(img, (299, 299)) for img in X])

    return np.repeat(np.expand_dims(X, -1), 3, axis=-1)


def preprocess_2(PDFs, threshold=1e-10, downsample_1=1, downsample_2=1, shape=(299, 299, 3)):
    """
    Preprocess PDFs:
        1. Take every ``downsample_1`` entry across axis 1 and every ``downsample_2`` entry across axis 2.
        2. Compute the maximal_value_in_datapoint for each PDF in PDFs.
        3. X = X / maximal_value_in_datapoint. Now, maximal value is 1.
        4. Everything below ``threshold`` is set to ``threshold``.
        5. X = np.log10(X)
        6. X = X + (-np.log10(threshold) / 2)
        7. X = X / (-np.log10(threshold) / 2)
        8. Resize to shape (PDFs.shape[0], 299, 299, nchannels) where channels are copied from one channel initially.
    Modifies PDFs in-place for memory efficiency.

    :param PDFs: Pictures with probability distributions (in the linear scale).
    :param threshold: Lower bound for the signal in units of the maximal signal.
    :param downsample_1: Take every ``downsample_1`` entry across axis 1.
    :param downsample_2: Take every ``downsample_2`` entry across axis 2.
    :param shape: Shape of the final picture. If more than one color channel, the values are repeated across channels.

    :return: preprocessed PDFs of shape (PDFs.shape[0], *shape).
    """
    X = PDFs[:, ::downsample_1, ::downsample_2].view()
    X /= np.max(X, axis=(1, 2)).reshape((-1, 1, 1))

    def flat_for(a, f, batch_size=8192):  # for in-place modification of np.ndarray
        a = a.reshape(-1)
        for i in range(0, a.shape[0], batch_size):
            a[i:i + batch_size] = f(a[i:i + batch_size])

    # X[np.where(X < threshold)] = threshold
    np.clip(X, threshold, None, out=X)
    # in the comments we use value threshold=1e-6 for illustration
    # X = np.log10(X)  # X in -6...0
    # flat_for(X, np.log10, batch_size=8192)
    np.log10(X, out=X)
    X += (-np.log10(threshold) / 2)  # X in -3...3
    X /= (-np.log10(threshold) / 2)  # X in -1...1

    X_resized = np.empty((X.shape[0], *shape[:2]), dtype=X.dtype)
    for i, img in enumerate(X):
        X_resized[i] = resize(img, shape[:2])
    return np.repeat(np.expand_dims(X_resized, -1), shape[2], axis=-1)


def preprocess_2_safe(PDFs, threshold=1e-10, downsample_1=1, downsample_2=1, shape=(299, 299, 3)):
    """
    Preprocess PDFs:
        1. Take every ``downsample_1`` entry across axis 1 and every ``downsample_2`` entry across axis 2.
        2. Compute the maximal_value_in_datapoint for each PDF in PDFs.
        3. X = X / maximal_value_in_datapoint. Now, maximal value is 1.
        4. Everything below ``threshold`` is set to ``threshold``.
        5. X = np.log10(X)
        6. X = X + (-np.log10(threshold) / 2)
        7. X = X / (-np.log10(threshold) / 2)
        8. Resize to shape (PDFs.shape[0], 299, 299, nchannels) where channels are copied from one channel initially.
    Does not modify PDFs in-place.

    :param PDFs: Pictures with probability distributions (in the linear scale).
    :param threshold: Lower bound for the signal in units of the maximal signal.
    :param downsample_1: Take every ``downsample_1`` entry across axis 1.
    :param downsample_2: Take every ``downsample_2`` entry across axis 2.
    :param shape: Shape of the final picture. If more than one color channel, the values are repeated across channels.

    :return: preprocessed PDFs of shape (PDFs.shape[0], *shape).
    """
    X = PDFs[:, ::downsample_1, ::downsample_2]
    X = X / np.max(X, axis=(1, 2)).reshape((-1, 1, 1))

    X[np.where(X < threshold)] = threshold

    # in the comments we use value threshold=1e-6 for illustration
    X = np.log10(X)  # X in -6...0
    X += (-np.log10(threshold) / 2)  # X in -3...3
    X /= (-np.log10(threshold) / 2)  # X in -1...1

    X_resized = np.empty((X.shape[0], *shape[:2]), dtype=X.dtype)
    for i, img in enumerate(X):
        X_resized[i] = resize(img, shape[:2])
    return np.repeat(np.expand_dims(X_resized, -1), shape[2], axis=-1)


def remove_region_around_origin(X, removal_size=(0.1, 0.2)):
    idxs1_min = int(X.shape[1] / 2 * (1 - removal_size[0] / 2))
    idxs1_max = int(X.shape[1] / 2 * (1 + removal_size[0] / 2))
    idxs2_min = 0
    idxs2_max = int(X.shape[2] * removal_size[1])
    X[:, idxs1_min:idxs1_max][:, :, idxs2_min:idxs2_max] = 0.
    return X


def get_spacing(grid):
    delta_z = grid[1, 0, 0] - grid[0, 0, 0]
    delta_x = grid[0, 1, 1] - grid[0, 0, 1]

    return delta_z, delta_x


def normalize(PDFs, delta_z, delta_x):
    norm = np.sum(np.sum(PDFs, axis=2), axis=1) * delta_z * delta_x

    return PDFs / norm.reshape(PDFs.shape[0], 1, 1)


def poisson_noise_for_preprocess_2(X, threshold, max_counts_per_pixel):
    X = np.power(10., 0.5 * np.log10(threshold) * (1 - X))  # X from 0 to 1 in linear scale
    X = (np.random.poisson(lam=X * max_counts_per_pixel, size=X.shape) + 1.) / max_counts_per_pixel
    X = np.log10(X)
    X = X + (-np.log10(threshold) / 2.)
    X = X / (-np.log10(threshold) / 2.)
    return X


def normalize_images(X):
    means = np.mean(X, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
    variances = np.std(X, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
    X -= means
    X /= 2.0 * variances
    return X


def detector_saturation(X, level=0.5):
    X = np.clip(X, -1., level)
    X = X + 1. - level
    return X
