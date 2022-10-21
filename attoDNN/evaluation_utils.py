import os

import numpy as np
import tensorflow as tf
from .attodataset import AttoDataset
from typing import List
from .train_utils import MAE
import warnings
from matplotlib import pyplot as plt
from time import time


def parse_model_filename(fn):
    bn = os.path.basename(fn)
    train_dataset_name, model_name, model_number = bn.split('__')
    model_number = int(model_number[:-3])
    return train_dataset_name, model_name, model_number


def evaluation_grid(models_filenames: List[str], AttoDatasetsList: List[AttoDataset], batch_size=32,
                    use_data_generator=False):
    tf.compat.v1.disable_eager_execution()  # for OOM issues in loops
    warnings.warn('TF: Disabled eager execution for evaluation in loop as a workaround for OOM issues.')

    grid = {}
    unique_model_names = []
    unique_train_dataset_names = []

    for i, fn in enumerate(models_filenames):
        model = tf.keras.models.load_model(fn)
        train_dataset_name, model_name, model_number = parse_model_filename(fn)

        if train_dataset_name not in grid.keys():
            grid[train_dataset_name] = {}
        if model_name not in grid[train_dataset_name].keys():
            grid[train_dataset_name][model_name] = {}
        if model_name not in unique_model_names:
            unique_model_names.append(model_name)
        if train_dataset_name not in unique_train_dataset_names:
            unique_train_dataset_names.append(train_dataset_name)

        for j, dataset in enumerate(AttoDatasetsList):
            current = i*len(AttoDatasetsList) + j
            total = len(AttoDatasetsList) * len(models_filenames)
            print(f'{current}/{total} {train_dataset_name} {model_name} {model_number} {dataset.basename}')

            if dataset.basename not in grid[train_dataset_name][model_name].keys():
                grid[train_dataset_name][model_name][dataset.basename] = {}

            if dataset.data_generator_test is None or not use_data_generator:
                X, y_true = dataset.get_Xy()
                y_pred = model.predict(X, batch_size=batch_size)
            else:
                identity_model = tf.keras.Sequential()
                identity_model.add(tf.keras.layers.Activation('linear'))
                y_true = identity_model.predict(dataset.data_generator_test, batch_size=batch_size)
                y_pred = model.predict(dataset.data_generator_test, batch_size=batch_size)

            grid[train_dataset_name][model_name][dataset.basename][model_number] = {'y_true': y_true,
                                                                                    'y_pred': y_pred,
                                                                                    'MAE': MAE(y_true, y_pred)}

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        del model

    return grid, unique_model_names, unique_train_dataset_names


def plot_evaluation_grid(grid: dict, model_names, train_dataset_names, eval_dataset_names,
                         subplot_kwargs=None, color_list=None, marker_list=None,
                         plot_single_model_number=None, max_points_per_dataset=None,
                         suptitle=None,
                         parse_dataset_name_fun=lambda x: x):
    if not isinstance(subplot_kwargs, dict):
        fig, axs = plt.subplots(len(train_dataset_names), len(model_names), squeeze=False)
    else:
        fig, axs = plt.subplots(len(train_dataset_names), len(model_names), squeeze=False, **subplot_kwargs)
    if color_list is None:
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if marker_list is None:
        marker_list = ['o', 's', 'v', 'D', '^', '<', '>']

    for i, train_dataset_name in enumerate(train_dataset_names):
        axs[i][0].set_ylabel(parse_dataset_name_fun(train_dataset_name).replace('_', '\n'), fontsize=12)
        if train_dataset_name in grid.keys():
            for j, model_name in enumerate(model_names):
                axs[0][j].set_title(model_name)

                if model_name in grid[train_dataset_name].keys():
                    for k, eval_dataset_name in enumerate(eval_dataset_names):
                        if eval_dataset_name in grid[train_dataset_name][model_name].keys():

                            result_model = grid[train_dataset_name][model_name][eval_dataset_name]

                            model_0 = next(iter(result_model.values()))
                            y_true = model_0['y_true']

                            n = 0
                            y_pred = np.zeros_like(y_true)
                            y_pred_std = np.zeros_like(y_true)
                            y_pred_all = []
                            if plot_single_model_number is None:
                                for model_number, result_model_number in result_model.items():
                                    y_pred_all.append(result_model_number['y_pred'])
                                    n += 1
                                if n > 0:
                                    y_pred_std = np.std(y_pred_all, axis=0)
                                    y_pred = np.mean(y_pred_all, axis=0)
                            else:
                                y_pred = result_model[plot_single_model_number]['y_pred']
                                n += 1

                            if max_points_per_dataset:
                                if y_pred.shape[0] > max_points_per_dataset:
                                    choice = np.random.choice(y_pred.shape[0], max_points_per_dataset, replace=False)
                                    y_true = y_true[choice]
                                    y_pred = y_pred[choice]
                                    y_pred_std = y_pred_std[choice]

                            axs[i][j].errorbar(y_true[:, 0], y_pred[:, 0], yerr=y_pred_std[:, 0],
                                               linestyle='none', marker=marker_list[k%len(marker_list)],
                                               markerfacecolor='none',
                                               label=parse_dataset_name_fun(eval_dataset_name)
                                               )
                            axs[i][j].annotate(f'${n=}$', (0.05, 0.95), xycoords='axes fraction')

                x = np.linspace(0, 1, 10)
                axs[i][j].plot(x, x, '-k')
                axs[i][j].set_aspect(1)

    for ax in axs[-1]:
        ax.set_xlabel('true')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1,1))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(suptitle)

    return fig, axs












