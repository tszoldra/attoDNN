import sys
import os
from functools import partial

import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import glob

from attoDNN.attodataset import AttoDataset
import attoDNN.data_generator as dg
import attoDNN.train_utils as tu
import attoDNN.preprocess as pp
import attoDNN.evaluation_utils as eu


import argparse
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a list of models on selected datasets.')
    parser.add_argument('datasets_list_txt', type=str, help='TXT List of datasets to evaluate on. '
                                                            'If the model was trained on specific dataset,'
                                                            'evaluation on this dataset is split into _training '
                                                            'and test (no extra suffix) parts.')
    parser.add_argument('models_path', type=str, help='*.h5 keras models.')
    parser.add_argument('output_pickle', type=str, help='Path to save pickle results of evaluation.', default='.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for predictions.')
    parser.add_argument('--batch_size', type=int, help='Batch size for predictions. Decrease if OOM errors occur.',
                        default=10)
    parser.add_argument('--save_plot', action='store_true', help='Whether to save plot with pred-true values.')
    parser.add_argument('--only_test', action='store_true', help='Predict only on the test data disregarding train data.')

    args = parser.parse_args()

    with open(args.datasets_list_txt, 'r') as f:
        datasets_paths = f.read().split('\n')[:-1]

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # =========== preprocessing ===========
    # same as in train.py used for training

    feature = 'Up'

    preprocess_kwargs = {
        'threshold': 1e-6,
        'downsample_1': 1,
        'downsample_2': 1,
        'shape': (224, 224, 1)
    }


    def preprocessor(PDFs):
        return pp.preprocess_2(PDFs, **preprocess_kwargs)


    # =========== dataset split ===========

    train_test_split_kwargs = {
        'val_size': 0.1,
        'test_size': 0.1,
        'random_state': 1234,
        'random_state_validation': 1234,
    }

    train_test_split = partial(tu.regression_train_test_split_4, **train_test_split_kwargs)

    datasets = [AttoDataset(fn) for fn in datasets_paths]

    for ds in datasets:
        ds.preprocess(preprocessor, feature, delete_NPZ=False)


    models = glob.glob(args.models_path)
    print('Models to analyze:')
    print(models)

    grid, unique_model_names, unique_train_dataset_names = eu.evaluation_grid(models, datasets,
                                                                              batch_size=args.batch_size,
                                                                              use_data_generator=False,
                                                                              train_test_split=train_test_split,
                                                                              predict_on_training_data=(not args.only_test))

    unique_eval_dataset_names = []
    for _, v in grid.items():
        for _, vv in v.items():
            for n, _ in vv.items():
                if n not in unique_eval_dataset_names:
                    unique_eval_dataset_names.append(n)

    with open(args.output_pickle, 'wb') as f:
        pickle.dump((grid, unique_model_names,
                     unique_train_dataset_names,
                     unique_eval_dataset_names), f)
        print(f'Analysis saved to {args.output_pickle}.')


    if args.save_plot:
        fig, axs = eu.plot_evaluation_grid(grid,
                                           unique_model_names,
                                           unique_train_dataset_names,
                                           eval_dataset_names=unique_eval_dataset_names,
                                           subplot_kwargs={'sharex': True, 'sharey': True, 'figsize': (12, 12), },
                                           color_list=None,
                                           marker_list=None,
                                           plot_single_model_number=None,
                                           max_points_per_dataset=10000,
                                           suptitle=args.models_path,
                                           parse_dataset_name_fun=lambda x: x)
        fig.savefig(args.output_pickle + '__gridplot.pdf', bbox_inches='tight')
        print(f'Plot saved to {args.output_pickle}__gridplot.pdf.')


