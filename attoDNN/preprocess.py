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



def preprocess_2(PDFs, threshold=1e-10, downsample_1=1, downsample_2=1):
    """
    Preprocess PDFs:
        1. Take every ``downsample_1`` entry across axis 1 and every ``downsample_2`` entry across axis 2.
        2. Compute the maximal_value_in_datapoint for each PDF in PDFs.
        3. X = X / maximal_value_in_datapoint. Now, maximal value is 1.
        4. Everything below ``threshold`` is set to ``threshold``.
        5. X = np.log10(X)
        6. X = X + (-np.log10(threshold) / 2)
        7. X = X / (-np.log10(threshold) / 2)
        8. Resize to shape (PDFs.shape[0], 299, 299, 3) where channels are copied from one channel initially.

    :param PDFs: Pictures with probability distributions (in the linear scale).
    :param threshold: Lower bound for the signal in units of the maximal signal.
    :param downsample_1: Take every ``downsample_1`` entry across axis 1.
    :param downsample_2: Take every ``downsample_2`` entry across axis 2.
    :return: function that acts on PDFs and returns shape (PDFs.shape[0], 299, 299, 3).
    """
    X = PDFs[:, ::downsample_1, ::downsample_2]

    X = X / np.max(X, axis=(1, 2)).reshape((-1, 1, 1))
    X[np.where(X < threshold)] = threshold
    # in the comments we use value threshold=1e-6 for readability
    X = np.log10(X)  # X in -6...0
    X = X + (-np.log10(threshold) / 2)  # X in -3...3
    X = X / (-np.log10(threshold) / 2)  # X in -1...1

    X = np.array([resize(img, (299, 299)) for img in X])

    return np.repeat(np.expand_dims(X, -1), 3, axis=-1)


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
