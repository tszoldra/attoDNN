import os.path
import numpy as np
from .preprocess import normalize, get_spacing
from .data_generator import DataGenerator

from functools import partial


class AttoDataset:

    def __init__(self, filename_npz: str):
        self.X = None
        self.y = None
        self.preprocessed = False

        self.NPZ = np.load(filename_npz)
        self.filename_npz = filename_npz
        self.basename = os.path.basename(filename_npz)

        self.PDFs = self.NPZ['PDFs'][:]
        self.labels = self.NPZ['labels']
        self.grid = self.NPZ['grid']
        self.fd = self.__feature_dict(filename_npz)
        self.delta_z, self.delta_x = get_spacing(self.grid)
        self.normalize_fun = partial(normalize, delta_z=self.delta_z, delta_x=self.delta_x)
        self.data_generator_train = None
        self.data_generator_val = None
        self.data_generator_test = None


    def __feature_dict(self, filename_npz):  # column headers for labels
        if 'QProp' in filename_npz:
            fd = {'Up': 0, 'N': 1, 'CEP': 2}
        elif 'SFA' in filename_npz:
            fd = {'Ip': 0, 'Up': 1, 'omega': 2, 'N': 3, 'CEP': 4, 'Target': 5}
        elif 'Experiment' in filename_npz and 'fakeN' in filename_npz:  # workaround to be able to predict N for data that does not have this label
            fd = {'Up': 0, 'N': 1}
        elif 'Experiment' in filename_npz:
            fd = {'Up': 0}
        else:
            raise NotImplementedError(f'Readout of the dataset {filename_npz} not implemented.')

        return fd


    def preprocess(self, preprocessor, feature_name, delete_NPZ=False):
        self.X = preprocessor(self.PDFs)
        self.y = self.labels[:, self.fd[feature_name]:self.fd[feature_name]+1]
        self.preprocessed = True
        if delete_NPZ:
            del self.PDFs, self.labels, self.grid, self.NPZ
        pass


    def get_Xy(self):
        if self.preprocessed:
            return self.X, self.y
        else:
            raise RuntimeError('You must preprocess the data first.')


    def get_sample_closest_to(self, y_0):
        if self.preprocessed:
            idx = np.argmin(np.abs(self.y.flatten() - y_0))
            return self.X[idx], self.y[idx], idx
        else:
            raise RuntimeError('You must preprocess the data first.')


    def set_data_generator_train(self, data_generator: DataGenerator):
        self.data_generator_train = data_generator

    def set_data_generator_val(self, data_generator: DataGenerator):
        self.data_generator_val = data_generator

    def set_data_generator_test(self, data_generator: DataGenerator):
        self.data_generator_test = data_generator



