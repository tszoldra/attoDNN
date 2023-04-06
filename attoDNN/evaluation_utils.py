import os

import numpy as np
import tensorflow as tf
from .attodataset import AttoDataset
from typing import List
from .train_utils import MAE, MAPE, MSE, RegressionNLL
import warnings
from matplotlib import pyplot as plt
from time import time
from scipy.stats import norm


def parse_model_filename(fn):
    bn = os.path.basename(fn)
    train_dataset_name, model_name, model_number = bn.split('__')
    model_number = int(model_number[:-3])
    # if '.npz_' in train_dataset_name:
    #    train_dataset_name = train_dataset_name[:train_dataset_name.index(".npz_")+4]

    return train_dataset_name, model_name, model_number


def evaluation_grid(models_filenames: List[str], AttoDatasetsList: List[AttoDataset], batch_size=32,
                    use_data_generator=False,
                    train_test_split=None,
                    predict_on_training_data=True,
                    data_gen_fun=None,
                    num_data_gen_realizations=1):
    tf.compat.v1.disable_eager_execution()  # for OOM issues in loops
    warnings.warn('TF: Disabled eager execution for evaluation in loop as a workaround for OOM issues.')

    grid = {}
    unique_model_names = []
    unique_train_dataset_names = []

    for i, fn in enumerate(models_filenames):
        model = tf.keras.models.load_model(fn, compile=False)
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
            current = i * len(AttoDatasetsList) + j
            total = len(AttoDatasetsList) * len(models_filenames)
            print(f'{current}/{total} {train_dataset_name} {model_name} {model_number} {dataset.basename}')

            if dataset.basename not in grid[train_dataset_name][model_name].keys():
                if not use_data_generator:
                    grid[train_dataset_name][model_name][dataset.basename] = {}
                else:
                    grid[train_dataset_name][model_name][dataset.basename + '_aug'] = {}

            def grid_record(y_true, y_pred):
                r = {}
                if y_pred.shape[-1] == 2:  # mean variance for the NLL regression
                    r['sigmasq_pred'] = y_pred[:, 1:]
                    y_pred = y_pred[:, 0:1]

                r.update({'y_true': y_true,
                          'y_pred': y_pred,
                          'MAE': MAE(y_true, y_pred),
                          'MAPE': MAPE(y_true, y_pred),
                          'MSE': MSE(y_true, y_pred),
                          'RMSE': np.sqrt(MSE(y_true, y_pred))})

                return r

            def get_generated_data_prediction(X0, y0, data_gen, num_data_gen_realizations):
                # # very inefficient
                # print('================== using augmentation ==================')
                # X, y_true = [[] for _ in range(num_data_gen_realizations)], [[] for _ in
                #                                                              range(num_data_gen_realizations)]
                # dataset.set_data_generator_test(data_gen(X0, y0))
                # with tf.compat.v1.Session() as sess:
                #     for ii in range(num_data_gen_realizations):
                #         for jj in range(dataset.data_generator_test.__len__()):
                #             x, y = dataset.data_generator_test.__getitem__(jj)
                #             X[ii].append(x.eval(session=sess))
                #             y_true[ii].append(y)
                #         dataset.data_generator_test.on_epoch_end()
                #
                # X = np.array(X)
                # y_true = np.array(y_true)
                # y_true = y_true.reshape(-1, y_true.shape[-1])
                #
                # X = X.reshape(-1, *(X.shape[-3:]))
                #
                # y_pred = model.predict(X.reshape(-1, *(X.shape[-3:])), batch_size=batch_size)
                # y_pred = y_pred.reshape(-1, y_pred.shape[-1])

                print('================== using augmentation ==================')
                # very inefficient workaround for "not in graph" errors
                dg = data_gen(X0, y0)

                X0_aug = np.zeros((dg.__len__(), dg.batch_size, *(X0.shape[1:])))
                y_true = []
                y_pred = []

                for ii in range(num_data_gen_realizations):
                    print(f'random realization {ii + 1} out of {num_data_gen_realizations}')
                    with tf.compat.v1.Session() as sess:
                        for batch_id in range(dg.__len__()):
                            x, y = dg.__getitem__(batch_id)
                            # slow copy to numpy...
                            X0_aug[batch_id] = x.eval(session=sess)
                            y_true.append(y)

                    y_pred_aug = model.predict(X0_aug.reshape(-1, *X0_aug.shape[2:]), batch_size=dg.batch_size)

                    y_pred.append(y_pred_aug)

                    dg.on_epoch_end()

                y_true = np.array(y_true).reshape(-1, y0.shape[-1])
                y_pred = np.array(y_pred).reshape(-1, y_pred[0].shape[-1])

                return y_true, y_pred

            if train_test_split is not None and dataset.basename in train_dataset_name:
                X, y = dataset.get_Xy()
                X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
                X_train = np.concatenate((X_train, X_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)

                if data_gen_fun is None or not use_data_generator:
                    if predict_on_training_data:
                        if dataset.basename + '_train' not in grid[train_dataset_name][model_name].keys():
                            grid[train_dataset_name][model_name][dataset.basename + '_train'] = {}

                        y_train_pred = model.predict(X_train, batch_size=batch_size)
                        grid[train_dataset_name][model_name][dataset.basename + '_train'][model_number] = grid_record(
                            y_train, y_train_pred)

                    y_test_pred = model.predict(X_test, batch_size=batch_size)
                    grid[train_dataset_name][model_name][dataset.basename][model_number] = grid_record(y_test,
                                                                                                       y_test_pred)

                else:
                    data_gen = data_gen_fun(fn)

                    if predict_on_training_data:
                        if dataset.basename + '_train_aug' not in grid[train_dataset_name][model_name].keys():
                            grid[train_dataset_name][model_name][dataset.basename + '_train_aug'] = {}

                        y_train, y_train_pred = get_generated_data_prediction(X_train, y_train, data_gen,
                                                                              num_data_gen_realizations)
                        grid[train_dataset_name][model_name][dataset.basename + '_train_aug'][
                            model_number] = grid_record(y_train, y_train_pred)

                    y_test, y_test_pred = get_generated_data_prediction(X_test, y_test, data_gen,
                                                                        num_data_gen_realizations)

                    grid[train_dataset_name][model_name][dataset.basename + '_aug'][model_number] = grid_record(y_test,
                                                                                                                y_test_pred)
            else:
                if data_gen_fun is None or not use_data_generator:
                    X, y_true = dataset.get_Xy()
                    y_pred = model.predict(X, batch_size=batch_size)
                    grid[train_dataset_name][model_name][dataset.basename][model_number] = grid_record(y_true, y_pred)

                else:
                    X, y_true = dataset.get_Xy()
                    data_gen = data_gen_fun(fn)

                    y_true, y_pred = get_generated_data_prediction(X, y_true, data_gen, num_data_gen_realizations)

                    grid[train_dataset_name][model_name][dataset.basename + '_aug'][model_number] = grid_record(y_true,
                                                                                                                y_pred)

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        del model

    return grid, unique_model_names, unique_train_dataset_names


def ensemble_predict(ensemble_dict, single_model=None):
    model_0 = next(iter(ensemble_dict.values()))
    y_true = model_0['y_true']

    n = 0
    y_pred = np.zeros_like(y_true)
    y_pred_std = np.zeros_like(y_true)  # empirical stdev
    y_pred_all = []
    if single_model is None:
        for model_number, result_model_number in ensemble_dict.items():
            y_pred_all.append(result_model_number['y_pred'])
            n += 1
        if n > 0:
            y_pred_std = np.std(y_pred_all, axis=0)
            y_pred = np.mean(y_pred_all, axis=0)
    else:
        y_pred = ensemble_dict[single_model]['y_pred']
        n += 1

    return y_true, y_pred, y_pred_std, n


def ensembleNLL_predict(ensemble_dict, single_model=None):
    # https://arxiv.org/pdf/1612.01474.pdf

    model_0 = next(iter(ensemble_dict.values()))
    y_true = model_0['y_true']

    n = 0
    y_pred = np.zeros_like(y_true)
    y_pred_std = np.zeros_like(y_true)  # empirical stdev
    y_pred_all = []
    sigmasq_pred_all = []
    if single_model is None:
        for model_number, result_model_number in ensemble_dict.items():
            y_pred_all.append(result_model_number['y_pred'])
            sigmasq_pred_all.append(result_model_number['sigmasq_pred'])
            n += 1
        if n > 0:
            y_pred = np.mean(y_pred_all, axis=0)
            y_pred_std = np.sqrt(
                np.mean(sigmasq_pred_all, axis=0) + np.mean(np.array(y_pred_all) ** 2, axis=0) - np.array(y_pred) ** 2)
    else:
        y_pred = ensemble_dict[single_model]['y_pred']
        y_pred_std = np.sqrt(ensemble_dict[single_model]['sigmasq_pred'])
        n += 1

    return y_true, y_pred, y_pred_std, n


def regressionNLL_losses(y_true, y_pred, sigmasq_pred):
    """Computes the samplewise losses for y_true.
    For NLL loss uses sigmasq_pred.
    For other losses, uses point prediction y_pred.
    
    """
    losses = {}

    nll = lambda y_true, y_pred, sigmasq_pred: 0.5 * (
                np.log(sigmasq_pred[:, 0]) + (y_true[:, 0] - y_pred[:, 0]) ** 2 / (
        sigmasq_pred[:, 0]))  # samplewise loss

    nlls = nll(y_true, y_pred, sigmasq_pred)

    losses['NLLs'] = nlls
    losses['NLL'] = np.mean(nlls)

    AEs = np.abs(y_true - y_pred)
    losses['AEs'] = AEs
    losses['MAE'] = np.mean(AEs)

    SEs = (y_true - y_pred) ** 2
    losses['SEs'] = SEs
    losses['MSE'] = np.mean(SEs)
    losses['RMSE'] = np.sqrt(losses['MSE'])

    APEs = np.abs(y_true - y_pred) / y_true * 100
    losses['APEs'] = APEs
    losses['MAPE'] = np.mean(APEs)

    return losses


def regressionNLL_losses_with_uncertainty(y_true, y_pred, y_true_std, sigmasq_pred, n_samples=100):
    """Computes the samplewise losses for y_true that may be given with uncertainties y_true_std.
    For NLL loss uses sigmasq_pred.
    For other losses, uses point prediction y_pred and random-sampled y_true.
    
    """
    losses = {}

    def rep(x):
        x_rep = np.repeat([x], n_samples, axis=0).T
        if len(x_rep.shape) == 1:
            x_rep = np.expand_dims(x_rep, axis=0)
        return x_rep[0]

    y_true_rep = norm.rvs(loc=rep(y_true), scale=rep(y_true_std))
    if len(y_true_rep.shape) == 1:  # happens for a single datapoint
        y_true_rep = np.expand_dims(y_true_rep, axis=0)

    y_pred_rep = rep(y_pred)
    sigmasq_pred_rep = rep(sigmasq_pred)

    nlls = 0.5 * np.mean((np.log(sigmasq_pred[:, :]) + (y_true[:, :] - y_pred[:, :]) ** 2 / (sigmasq_pred[:, :])),
                         axis=1)  # samplewise loss

    losses['NLLs'] = np.expand_dims(nlls, axis=-1)
    losses['NLL'] = np.mean(nlls)

    AEs = np.mean(np.abs(y_true_rep - y_pred_rep), axis=1)
    losses['AEs'] = np.expand_dims(AEs, axis=-1)
    losses['MAE'] = np.mean(AEs)

    SEs = np.mean((y_true_rep - y_pred_rep) ** 2, axis=1)
    losses['SEs'] = np.expand_dims(SEs, axis=-1)
    losses['MSE'] = np.mean(SEs)
    losses['RMSE'] = np.sqrt(losses['MSE'])

    APEs = np.mean(np.abs(y_true_rep - y_pred_rep) / y_true_rep, axis=1) * 100
    losses['APEs'] = np.expand_dims(APEs, axis=-1)
    APEs_std = np.std(np.abs(y_true_rep - y_pred_rep) / y_true_rep, axis=1) * 100
    losses['APEs_std'] = np.expand_dims(APEs_std, axis=-1)

    losses['MAPE'] = np.mean(APEs)

    return losses


def plot_evaluation_grid(grid: dict, model_names, train_dataset_names, eval_dataset_names,
                         subplot_kwargs=None, color_list=None, marker_list=None,
                         plot_single_model_number=None, max_points_per_dataset=None,
                         suptitle=None,
                         parse_dataset_name_fun=lambda x: x, ):
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

                            if 'Bayesian' in model_name:
                                y_true, y_pred, y_pred_std, n = ensembleNLL_predict(result_model,
                                                                                    single_model=plot_single_model_number)
                            else:
                                y_true, y_pred, y_pred_std, n = ensemble_predict(result_model,
                                                                                 single_model=plot_single_model_number)

                            if max_points_per_dataset:
                                if y_pred.shape[0] > max_points_per_dataset:
                                    choice = np.random.choice(y_pred.shape[0], max_points_per_dataset, replace=False)
                                    y_true = y_true[choice]
                                    y_pred = y_pred[choice]
                                    y_pred_std = y_pred_std[choice]

                            axs[i][j].errorbar(y_true[:, 0], y_pred[:, 0], yerr=y_pred_std[:, 0],
                                               linestyle='none', marker=marker_list[k % len(marker_list)],
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
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(suptitle)

    return fig, axs
