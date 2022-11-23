import os
from time import time
import keras.callbacks
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf
import argparse
import io

import matplotlib.pyplot as plt


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def set_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def to_accumulative(orig_optimizer, update_params_frequency, accumulate_sum_or_mean=True):
    # https://github.com/keras-team/keras/issues/3556
    if update_params_frequency < 1:
        raise ValueError('update_params_frequency must be >= 1')
    #print('update_params_frequency: %s' % update_params_frequency)
    #print('accumulate_sum_or_mean: %s' % accumulate_sum_or_mean)
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    accumulated_iterations = K.variable(0, dtype='int64', name='accumulated_iterations')
    orig_optimizer.accumulated_iterations = accumulated_iterations

    def updated_get_gradients(self, loss, params):
        return self.accumulate_gradient_accumulators

    def updated_get_updates(self, loss, params):
        self.accumulate_gradient_accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        updates_accumulated_iterations = K.update_add(accumulated_iterations, 1)
        new_grads = orig_get_gradients(loss, params)
        if not accumulate_sum_or_mean:
            new_grads = [g / K.cast(update_params_frequency, K.dtype(g)) for g in new_grads]
        self.updated_grads = [K.update_add(p, g) for p, g in zip(self.accumulate_gradient_accumulators, new_grads)]
        def update_function():
            with tf.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p))) for p in self.accumulate_gradient_accumulators]
            return tf.group(*(reset_grads + [updates_accumulated_iterations]))
        def just_store_function():
            return tf.group(*[updates_accumulated_iterations])
        
        update_switch = K.equal((updates_accumulated_iterations) % update_params_frequency, 0)
        
        with tf.control_dependencies(self.updated_grads):
            self.updates = [K.switch(update_switch, update_function, just_store_function)]
            return self.updates

    orig_optimizer.get_gradients = updated_get_gradients.__get__(orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = updated_get_updates.__get__(orig_optimizer, type(orig_optimizer))


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

    training_idxs, val_idxs = train_test_split(training_idxs, test_size=val_size * (1 - test_size),
                                               random_state=random_state)

    X_train = X[training_idxs]
    y_train = y[training_idxs]

    X_val = X[val_idxs]
    y_val = y[val_idxs]

    X_test = X[test_idxs]
    y_test = y[test_idxs]

    return X_train, X_val, X_test, y_train, y_val, y_test


def lr_scheduler(decay_rate=0.5, decay_step=80):
    """
    Learning rate scheduler. Every ``decay_step`` epochs, the learning rate decreases by a factor of ``decay_rate``.
    For use with ``keras.callbacks.LearningRateScheduler``.

    :param decay_rate: Decay rate.
    :param decay_step: Decay step.
    :returns: function that is the learning rate scheduler
    """
    def f(epoch, lr):
        if epoch % decay_step == 0 and epoch:
            return lr * pow(decay_rate, np.floor(epoch / decay_step))
        return lr
    return f


def dict_to_npz_dict(d):
    d1 = {}
    for key, val in d.items():
        d1[key] = np.array(val)
    return d1


# class ExtraValidation(keras.callbacks.Callback):
#     """Log evaluation metrics of an extra validation set. This callback
#     is useful for model training scenarios where multiple validation sets
#     are used for evaluation (as Keras by default, provides functionality for
#     evaluating on a single validation set only).
#     The evaluation metrics are also logged to TensorBoard.
#     Adapted from https://github.com/tanzhenyu/image_augmentation/blob/master/image_augmentation/callbacks/extra_eval.py
#     Args:
#         validation_data: A (X, y) dataset used to evaluate the
#             model, essentially an extra validation dataset.
#         validation_freq: Number of epochs to wait before performing
#             subsequent evaluations.
#     """
#     def __init__(self, X_val, y_val, validation_freq=1):
#         super(ExtraValidation, self).__init__()
#
#         self.X_val = X_val
#         self.y_val = y_val
#         self.validation_freq = validation_freq
#
#     def on_epoch_end(self, epoch, logs=None):
#         # evaluate at an interval of `validation_freq` epochs
#         if (epoch + 1) % self.validation_freq == 0:
#             # TODO: fix `model.evaluate` memory leak on TPU
#             # gather the evaluation metrics
#             scores = self.model.evaluate(self.X_val, self.y_val, verbose=2)

class ExtraValidation(keras.callbacks.Callback):
    """Log evaluation metrics of an extra validation set. This callback
    is useful for model training scenarios where multiple validation sets
    are used for evaluation (as Keras by default, provides functionality for
    evaluating on a single validation set only).
    The evaluation metrics are also logged to TensorBoard.
    Taken from https://github.com/tanzhenyu/image_augmentation/blob/master/image_augmentation/callbacks/extra_eval.py
    Args:
        validation_data: A tf.data.Dataset pipeline used to evaluate the
            model, essentially an extra validation dataset.
        tensorboard_path: Path to the TensorBoard logging directory.
        validation_freq: Number of epochs to wait before performing
            subsequent evaluations.
    """
    def __init__(self, X_val, y_val, tensorboard_path, id: str, validation_freq=1, plot=True):
        super(ExtraValidation, self).__init__()

        self.X_val = X_val
        self.y_val = y_val
        self.id = id  # identifier of the validation dataset for tensorboard
        self.tensorboard_path = tensorboard_path

        self.tensorboard_writer = tf.summary.create_file_writer(self.tensorboard_path)

        self.validation_freq = validation_freq

    def on_epoch_end(self, epoch, logs=None):
        # evaluate at an interval of `validation_freq` epochs
        if (epoch + 1) % self.validation_freq == 0:
            # gather metric names form model
            metric_names = ['{}'.format('epoch', metric.name)
                            for metric in self.model.metrics]

            # TODO: fix `model.evaluate` memory leak on TPU
            # gather the evaluation metrics
            scores = self.model.evaluate(self.X_val, self.y_val, verbose=2)
            y_pred = self.model.predict(self.X_val, verbose=0)

            fig, ax = plt.subplots(figsize=(6, 6))
            if y_pred.shape[-1] == 1:
                ax.scatter(np.array(self.y_val)[:, 0], np.array(y_pred)[:, 0])
            else:
                ax.errorbar(np.array(self.y_val)[:, 0], np.array(y_pred)[:, 0], yerr=np.array(y_pred)[:, 1])
            ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '-k')
            ax.set_xlabel('true label')
            ax.set_ylabel('predicted label')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            img = plot_to_image(fig)

            # gather evaluation metrics to TensorBoard
            with self.tensorboard_writer.as_default():
                #for metric_name, score in zip(metric_names, [scores]):
                tf.summary.scalar(metric_names[0] + " extra validation " + self.id, scores, step=epoch)
                tf.summary.image(f'Correlation plot {self.id}', img, step=epoch)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image




def model_compile_train_save(dg_train, dg_val, dg_test,
                             model_fun, epochs, optimizer,
                             loss, callbacks,
                             model_save_filename,
                             log_filename,
                             checkpoint=False,
                             checkpoint_filename=None,
                             fine_tune=True,
                             optimizer_fine_tune=None,
                             kernel_regularizer=None,
                             epochs_fine_tune=None,
                             callbacks_fine_tune=None,
                             ):

    model, base_model = model_fun(dg_train.input_dim, dg_train.output_dim)

    if kernel_regularizer is not None:
        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, kernel_regularizer())

    model.compile(optimizer=optimizer(), loss=loss())

    def get_cb(cb_fun):
        if checkpoint:
            if not checkpoint_filename:
                raise ValueError("Need to set checkpoint_filename.")

            cb = cb_fun() + [keras.callbacks.ModelCheckpoint(
                                                            filepath=checkpoint_filename,
                                                            save_weights_only=True,
                                                            monitor='val_loss',
                                                            mode='min',
                                                            save_best_only=True)]
        else:
            cb = cb_fun()
        return cb

    cb = get_cb(callbacks)

    start = time()

    history = model.fit(dg_train, validation_data=dg_val, epochs=epochs, callbacks=cb, verbose=2)
    if checkpoint:
        model.load_weights(checkpoint_filename)
    
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    
    # fine-tune the pretrained part if it exists
    if base_model and fine_tune:
        if optimizer_fine_tune is None:
            optimizer_fine_tune = optimizer()
        base_model.trainable = True
        model.compile(optimizer=optimizer_fine_tune(), loss=loss())

        if epochs_fine_tune is None:
            epochs_fine_tune = epochs
        if callbacks_fine_tune is None:
            cb_fine_tune = get_cb(callbacks)
        else:
            cb_fine_tune = get_cb(callbacks_fine_tune)

        history_fine_tuning = model.fit(dg_train, validation_data=dg_val, epochs=epochs_fine_tune,
                                        callbacks=cb_fine_tune, verbose=2)

        if checkpoint:
            model.load_weights(checkpoint_filename)
        
        train_loss = history_fine_tuning.history['loss'][-1]
        val_loss = history_fine_tuning.history['val_loss'][-1]
        

    end = time()
    
    model.save(model_save_filename)

    test_loss = model.evaluate(dg_test, verbose=2)

    with open(log_filename, 'a') as logs:
        if os.stat(log_filename).st_size == 0:
            logs.write('filename train_loss val_loss test_loss total_training_time[h]\n')
        logs.write(os.path.basename(model_save_filename) + f' {train_loss: .4f} {val_loss: .4f} {test_loss: .4f} {(end-start)/3600.: .1f}\n')
        logs.flush()



    np.savez(model_save_filename[:-3] + "___history.npz", **dict_to_npz_dict(history.history))
    if base_model and fine_tune:
        np.savez(model_save_filename[:-3] + "___history_fine_tuning.npz", **dict_to_npz_dict(history_fine_tuning.history))
    
    return model


class RegressionNLL(tf.keras.losses.Loss):
    # https://stackoverflow.com/questions/60385762/unable-to-get-good-results-when-trying-to-predict-mean-as-well-as-standard-devia
    # https://github.com/tensorflow/tensorflow/issues/39702
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)

    def call(self, y_true, y_pred):
        return 0.5 * K.mean(K.log(y_pred[:, 1] + self.epsilon) + K.square(y_true - y_pred[:, 0]) / (y_pred[:, 1] + self.epsilon))


def train_parser():
    parser = argparse.ArgumentParser(description='Train a model on dataset with given hyperparameters')
    myargs = [
        ('model_save_folder', str, 'Where to save models and logs.', '.'),
        ('dataset_path', str, 'Training dataset path.',
         '/nfs/amd0/home/c8888/attosecond_ML/data_cleaned/cartesian_0/QProp_unnormalized/QProp_Ar_LongPulse_CEP2.npz'),
        ('random_state', int, 'Random state for train-test split etc.', 1234),
        ('n_models', int, 'Total number of instances of a model to train.', 1),
        ('model_name', str, 'Model name.', 'VGG16'),
        ('checkpoint_filename', str, 'Path to checkpoint file.', '/home/c8888/scratch/checkpoint/checkpoint'),
        ('saturation_level_min', float, 'Minimal (worst-case) detector saturation level, in the range (-1, 1).', -0.5),
        ('saturation_level_max', float, 'Maximal (best-case) detector saturation level, in the range (-1, 1).', 1.),
        ('batch_size', int, 'Batch size. Should be chosen depending on available memory and model size.', 32),

    ]

    for argname, argtype, arghelp, argdefault in myargs:
        parser.add_argument(argname, type=argtype, help=arghelp, default=argdefault)

    parser.add_argument('--dense_layers', type=int, help='List of sizes of dense layers applied at the end of the network. If not given,'
                                 ' a single layer with linear activation is applied.',
                        default=None,
                        nargs='+')
    parser.add_argument('--tensorboard_path', type=str, help='Path to save tensorboard logs.',
                        default='./tensorboard')

    return parser
