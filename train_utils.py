import os
from time import time
import keras.callbacks
import numpy as np
from sklearn.model_selection import train_test_split



def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def regression_train_test_split_1(X, y, d=4, test_size=0.5, random_state=1234):
    """
    Use every d-th unique value of y for validation/testing.

    Arguments:
    :param X: Values of independent variable eg. pictures.
    :param y: Labels.
    :param d: Every ``d``-th value is taken out of training set.
    :param test_size: Fraction of the dataset made by taking every ``d``-th value out of training dataset that
        is used as testing dataset.
    :param random_state: Random seed used for validation/test splitting.

    :returns: (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    unique_y = np.sort(np.unique(y.flatten()))
    all_idxs = np.arange(y.shape[0])

    training_idxs = all_idxs[~np.isin(y.flatten(), unique_y[::d])]
    test_idxs = all_idxs[np.isin(y.flatten(), unique_y[::d])]

    val_idxs, test_idxs = train_test_split(test_idxs, test_size=test_size, random_state=random_state)

    X_train = X[training_idxs]
    y_train = y[training_idxs]

    X_val = X[val_idxs]
    y_val = y[val_idxs]

    X_test = X[test_idxs]
    y_test = y[test_idxs]

    return X_train, X_val, X_test, y_train, y_val, y_test


def regression_train_test_split_2(X, y, val_size=0.1, test_size=0.1, random_state=1234):
    """
    Validation/testing values of y are selected randomly in a standard way.

    Arguments:
    :param X: Values of independent variable eg. pictures.
    :param y: Labels.
    :param val_size: Size of the validation dataset as a fraction of the whole dataset.
    :param test_size: Size of the test dataset as a fraction of the whole dataset.
    :param random_state: Random seed used for train/(validation+test) and validation/test splitting.

    :returns: (X_train, X_val, X_test, y_train, y_val, y_test)

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=val_size + test_size,
                                                        random_state=random_state)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_size / (val_size + test_size),
                                                    random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def regression_train_test_split_3(X, y, val_size=0.1, test_size=0.1, random_state=1234):
    """
    Take out fraction test_size central values of the label for testing and validation.

    Arguments:
    :param X: Values of independent variable eg. pictures.
    :param y: Labels.
    :param val_size: Fraction of the whole dataset to use for validation.
    :param test_size: Fraction of the whole dataset that is taken out for testing (central values
    of the label). This range of parameters is also never used for validation.
    :param random_state: Random seed used for validation/test splitting.

    :returns: (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    unique_y = np.sort(np.unique(y.flatten()))
    all_idxs = np.arange(y.shape[0])

    L = unique_y.shape[0]
    cond_test = np.isin(y.flatten(), unique_y[int(L/2 - test_size * L/2):int(L/2 + test_size * L/2)])
    test_idxs = all_idxs[cond_test]
    training_idxs = all_idxs[~cond_test]

    training_idxs, val_idxs = train_test_split(test_idxs, test_size=val_size * (1 - test_size),
                                               random_state=random_state)

    X_train = X[training_idxs]
    y_train = y[training_idxs]

    X_val = X[val_idxs]
    y_val = y[val_idxs]

    X_test = X[test_idxs]
    y_test = y[test_idxs]

    return X_train, X_val, X_test, y_train, y_val, y_test


def lr_scheduler(epoch, lr, decay_rate=0.5, decay_step=80):
    """
    Learning rate scheduler. Every ``decay_step`` epochs, the learning rate decreases by a factor of ``decay_rate``.
    For use with ``keras.callbacks.LearningRateScheduler``.

    :param epoch: Current epoch.
    :param lr: Learning rate.
    :param decay_rate: Decay rate.
    :param decay_step: Decay step.
    :returns: lr
    """
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr


def model_compile_train_save(data_generators,
                             model_fun, epochs, optimizer,
                             loss, callbacks,
                             model_save_filename,
                             log_filename,
                             checkpoint=False,
                             checkpoint_filename=None,
                             fine_tune=True,
                             optimizer_fine_tune=None,
                             kernel_regularizer=None,
                             batch_size=32, batch_size_fine_tune=32,):

    dg_train, dg_val, dg_test = data_generators

    model, base_model = model_fun(dg_train.input_shape, dg_train.output_shape)

    if kernel_regularizer is not None:
        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, kernel_regularizer)

    model.compile(optimizer=optimizer, loss=loss)

    if checkpoint:
        if not checkpoint_filename:
            raise ValueError("Need to set checkpoint_filename.")

        callbacks = callbacks + [keras.callbacks.ModelCheckpoint(
                                                        filepath=checkpoint_filename,
                                                        save_weights_only=True,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_best_only=True)]

    start = time()

    history = model.fit(dg_train, batch_size=batch_size,
                        validation_data=dg_val, epochs=epochs, callbacks=callbacks)
    if checkpoint:
        model.load_weights(checkpoint_filename)

    # fine-tune the pretrained part if it exists
    if base_model and fine_tune:
        if optimizer_fine_tune is None:
            optimizer_fine_tune = optimizer
        base_model.trainable = True
        model.compile(optimizer=optimizer_fine_tune, loss=loss)

        history_fine_tuning = model.fit(dg_train, batch_size=batch_size_fine_tune,
                                        validation_data=dg_val, epochs=epochs, callbacks=callbacks)

        if checkpoint:
            model.load_weights(checkpoint_filename)

    end = time()

    model.save(model_save_filename)


    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    test_loss = model.evaluate(dg_test, batch_size=np.min([batch_size, batch_size_fine_tune]))

    with open(log_filename, 'a') as logs:
        if os.stat(log_filename).st_size == 0:
            logs.write('model_save_filename train_loss val_loss test_loss total_training_time[h]')
        logs.write(model_save_filename + f' {train_loss: .4f} {val_loss: .4f} {test_loss: .4f} {(end-start)/3600.: .1f}')
        logs.flush()

    def dict_to_npz_dict(d):
        d1 = {}
        for key, val in d.items():
            d1[key] = np.array(val)
        return d1

    np.savez(model_save_filename + "___history.npz", **dict_to_npz_dict(history.history))
    if base_model:
        np.savez(model_save_filename + "___history_fine_tuning.npz", **dict_to_npz_dict(history_fine_tuning.history))
