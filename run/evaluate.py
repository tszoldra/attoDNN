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
    parser.add_argument('--feature', type=str, default='Up', help='Feature to predict.', choices=['Up', 'N', 'CEP'])
    parser.add_argument('--threshold', type=float, default=1e-6, help='Threshold for lowest value in preprocessing.')
    parser.add_argument('--remove_region_around_origin', action='store_true', help='Whether to remove the region (rectangle) around origin. Size can be specified by frac_x_remove, frac_y_remove.')
    parser.add_argument('--frac_x_remove', type=float, default=0.5, help='Fraction of first axis to remove. Not applied if remove_region_around_origin not specified.')
    parser.add_argument('--frac_y_remove', type=float, default=0.3, help='Fraction of second axis to remove. Not applied if remove_region_around_origin not specified.')
    parser.add_argument('--use_data_generator', action='store_true', help='Whether to use the data generator (parameters except SL hardcoded in the source) to test the model. That is, the data is augmented also during testing phase which is a non-standard practice.')
    parser.add_argument('--saturation_level_min', type=float, default=1.0, help='Minimal (worst-case) detector saturation level, in the range (-1, 1). Used only if --use_data_generator is used.'),
    parser.add_argument('--num_data_gen_realizations', type=int, default=1, help='How many independent random realizations of the data generator are used. Used only if --use_data_generator is used.'),
    parser.add_argument('--auto_saturation_level_min', action='store_true', help='Automatic minimal (worst-case) detector saturation level, in the range (-1, 1). '
                                                                                 'Determined based on model filename which contains info about SL. Used only if --use_data_generator is used.'),
    args = parser.parse_args()


    with open(args.datasets_list_txt, 'r') as f:
        datasets_paths = f.read().split('\n')[:-1]

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # =========== preprocessing ===========
    # same as in train.py used for training

    feature = args.feature

    preprocess_kwargs = {
        'threshold': args.threshold,
        'downsample_1': 1,
        'downsample_2': 1,
        'shape': (224, 224, 1)
    }


    def preprocessor(PDFs):
        if args.remove_region_around_origin:
            return pp.remove_region_around_origin(pp.preprocess_2(PDFs, **preprocess_kwargs), removal_size=(args.frac_x_remove, args.frac_y_remove))
        else:
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

    if args.use_data_generator:
        def transform_X_fun(X, SL=1.0):
            X = dg.random_detector_saturation_tf(X,
                                                 saturation_level_min=SL,
                                                 saturation_level_max=1.0,
                                                 )
            X = dg.random_flip_lrud_tf(X)
            X = dg.random_contrast_tf(X, contrast_min=0.1, contrast_max=1.0)
            X = dg.random_brightness_tf(X, brightness_max_delta=1.0)
            X = tf.clip_by_value(X, -1, 1)
            return X

        def transform_y_fun(y):
            return y

        if not args.auto_saturation_level_min:
            def data_gen_fun(model_filename):
                return partial(dg.DataGenerator,  transform_X_fun=partial(transform_X_fun, SL=args.saturation_level_min),
                        transform_y_fun=transform_y_fun,
                        batch_size=8,
                        shuffle=True)
        else:
            def data_gen_fun(model_filename):
                def get_SL_from_filename(fn):
                    tab = fn.split('_')
                    SL = None
                    for i in range(len(tab)):
                        if tab[i] == 'SL':
                            SL = float(tab[i+1])
                    return SL

                SL = get_SL_from_filename(model_filename)
                print(f'for model {model_filename} using SL {SL}')
                return partial(dg.DataGenerator,  transform_X_fun=partial(transform_X_fun, SL=SL),
                        transform_y_fun=transform_y_fun,
                        batch_size=8,
                        shuffle=True)

    else:
        data_gen_fun = None



    models = glob.glob(args.models_path)
    print('Models to analyze:')
    print(models)

    grid, unique_model_names, unique_train_dataset_names = eu.evaluation_grid(models, datasets,
                                                                              batch_size=args.batch_size,
                                                                              use_data_generator=args.use_data_generator,
                                                                              train_test_split=train_test_split,
                                                                              predict_on_training_data=(not args.only_test),
                                                                              data_gen_fun=data_gen_fun,
                                                                              num_data_gen_realizations=args.num_data_gen_realizations)

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


