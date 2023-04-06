import os
import numpy as np
from functools import partial, lru_cache
from scipy import stats, integrate, interpolate
from attoDNN.attodataset import AttoDataset


def CEP_averaging(label, PDFs, labels, feature_dict):
    CEP_idx = feature_dict['CEP']
    not_CEP_idxs = np.where(np.arange(len(label)) != CEP_idx)

    not_CEP_equal = lambda x: np.allclose(x[not_CEP_idxs], label[not_CEP_idxs])

    other_CEP_idxs = np.where(np.array([not_CEP_equal(x) for x in labels]))

    return np.mean(PDFs[other_CEP_idxs], axis=0)


def dataset_to_CEP_averaged(ds: AttoDataset, out_filename_npz: str):
    CEP_idx = ds.fd['CEP']
    not_CEP_idxs = np.where(np.arange(ds.labels.shape[1]) != CEP_idx)

    unique_labels_without_CEP, idxs = np.unique(ds.labels[:, not_CEP_idxs], return_index=True, axis=0)

    PDFs_CEP_avg = np.zeros((idxs.shape[0], *ds.PDFs.shape[1:]))

    for i, idx in enumerate(idxs):
        if i % 100 == 0:
            print(f'{i} out of {len(idxs)}')
        label = ds.labels[idx]
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
