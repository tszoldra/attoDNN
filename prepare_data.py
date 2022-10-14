import os
import numpy as np
from functools import partial, lru_cache
from scipy import stats, integrate, interpolate
from attodataset import AttoDataset


@lru_cache(maxsize=2048)
def weight_tight_focusing(I0, I):
    return 1. / I * integrate.quad(lambda zeta: (1. + zeta ** 2) * np.sqrt(np.log(I0 / (I * (1. + zeta ** 2)))), 0.,
                                   np.sqrt(I0 / I - 1.))[0]


def weight_weak_focusing(I0, I):
    return 1. / I * np.sqrt(np.log(I0 / I))


def focal_averaging(label, PDFs, labels, feature_dict, weight_fun=weight_tight_focusing):
    # label is the current label, PDFs and labels are over the whole dataset
    # function assumes equal distances between Up datapoints

    Up_idx = feature_dict['Up']  # labels column with Up
    not_Up_idxs = np.where(
        np.arange(len(label)) != Up_idx)  # contains indexes of columns that are not Up but eg. CEP, N etc.

    Up0 = label[feature_dict['Up']]  # ponderomotive potential we are considering now. It is proportional to I.

    # if the lowest intensity in the dataset: nothing to average
    if Up0 == np.min(labels[:, Up_idx]):
        return PDFs

    not_Up_equal = lambda x: np.allclose(x[not_Up_idxs], label[
        not_Up_idxs])  # function that returns True if all other labels of the data entry except the Up have the same value

    Up_leq_Up0_idxs = np.where(np.logical_and(labels[:, Up_idx] <= Up0,
                                              np.array([not_Up_equal(x) for x in
                                                        labels])))  # data entries with lower intensities than Up0 and the same other labels

    Ups = labels[Up_leq_Up0_idxs][:, Up_idx]

    return np.einsum('i, ijk -> jk', np.array([weight_fun(Up0, x) for x in Ups]),
                     PDFs[Up_leq_Up0_idxs])  # result is not normalized


def dataset_to_focal_averaged(ds: AttoDataset, out_filename_npz: str, weight_fun=weight_tight_focusing):
    PDFs_FA = np.zeros_like(ds.PDFs)

    for i in range(ds.PDFs.shape[0]):
        label = ds.labels[i]
        if label[ds.fd['Up']] > np.min(ds.labels[:, ds.fd['Up']]):
            PDFs_FA[i] = focal_averaging(label, ds.PDFs, ds.labels, ds.fd, weight_fun=weight_fun)
        else:
            PDFs_FA[i] = ds.PDFs[i]

    np.savez(out_filename_npz, PDFs=PDFs_FA, labels=ds.labels, grid=ds.grid)


def CEP_averaging(label, PDFs, labels, feature_dict):
    #TODO test
    CEP_idx = feature_dict['CEP']
    not_CEP_idxs = np.where(np.arange(len(label)) != CEP_idx)

    not_CEP_equal = lambda x: np.allclose(x[not_CEP_idxs], label[not_CEP_idxs])


    other_CEP_idxs = np.where(np.array([not_CEP_equal(x) for x in labels]))

    return np.mean(PDFs[other_CEP_idxs], axis=0)


def dataset_to_CEP_averaged(ds: AttoDataset, out_filename_npz: str):
    CEP_idx = ds.fd['CEP']
    not_CEP_idxs = np.where(np.arange(ds.labels.shape[1]) != CEP_idx)

    unique_labels_without_CEP, idxs = np.unique(ds.labels[:, not_CEP_idxs], return_index=True, axis=0)


    PDFs_CEP_avg = np.zeros((idxs.shape[0], ds.PDFs.shape[1:]))

    for i in idxs:
        label = ds.labels[i]
        PDFs_CEP_avg[i] = CEP_averaging(label, ds.PDFs, ds.labels, ds.fd)

    np.savez(out_filename_npz, PDFs=PDFs_CEP_avg, labels=ds.labels[idxs], grid=ds.grid)


def intensity_averaging(relative_uncertainty, label, PDFs, labels, feature_dict):
    Up_idx = feature_dict['Up']
    not_Up_idxs = np.where(np.arange(len(label)) != Up_idx)

    Up0 = label[feature_dict['Up']]  # Up ~ I

    not_Up_equal = lambda x: np.allclose(x[not_Up_idxs], label[not_Up_idxs])

    other_Up_idxs = np.where(np.array([not_Up_equal(x) for x in labels]))

    PDFs_1 = PDFs[other_Up_idxs]

    sigma = relative_uncertainty * Up0
    weights = np.exp(-(Up0 - labels[other_Up_idxs][:, Up_idx]) ** 2 / sigma ** 2)
    return np.mean(PDFs_1 * weights.reshape((-1, 1, 1,)), axis=0)  # not normalized


def dataset_to_intensity_avg(ds: AttoDataset, out_filename_npz: str, relative_uncertainty, ):
    PDFs_intensity_avg = np.zeros_like(ds.PDFs)

    for i in range(ds.PDFs.shape[0]):
        label = ds.labels[i]
        if i % 100 == 0:
            print(f"{i} out of {ds.PDFs.shape[0]}")
        PDFs_intensity_avg[i] = intensity_averaging(relative_uncertainty, label, ds.PDFs, ds.labels, ds.fd)

    np.savez(out_filename_npz, PDFs=PDFs_intensity_avg, labels=ds.labels, grid=ds.grid)
