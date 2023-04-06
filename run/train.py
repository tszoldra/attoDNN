import sys
import os
from functools import partial
import numpy as np

import tensorflow as tf

from attoDNN.attodataset import AttoDataset
from attoDNN import nets, train_utils as tu, preprocess as pp, data_generator as dg
import datetime

if __name__ == "__main__":
    print(nets.list_GPUs())

    # tu.set_memory_growth()
    parser = tu.train_parser()
    args = parser.parse_args()

    # ============= CONFIGURATION ================
    model_save_folder = args.model_save_folder
    dataset_path = args.dataset_path

    datasets_extra_validation_paths = [
        '../../public_dataset/data_preprocessed/Experiment/Experiment_Argon_Intensity_Sweep_01_cutoff.npz',
        '../../public_dataset/data_preprocessed/Experiment//Experiment_Argon_Intensity_Sweep_02_cutoff.npz',
        '../../public_dataset/data_preprocessed/Experiment//Experiment_Argon_Intensity_Sweep_03_cutoff.npz',
        '../../public_dataset/data_preprocessed/Experiment//Experiment_Argon_Intensity_Sweep_04_cutoff.npz',
        '../../public_dataset/data_preprocessed/Experiment//Experiment_Argon_Single_Intensity_01_cutoff.npz',
    ]
    ds_extra_val = [AttoDataset(fn) for fn in datasets_extra_validation_paths]

    # =========== preprocessing ===========

    feature = 'Up'

    preprocess_kwargs = {
        'threshold': 1e-6,
        'downsample_1': 1,
        'downsample_2': 1,
        'shape': (224, 224, 1)
    }

    def preprocessor(PDFs):
        return pp.preprocess_2(PDFs, **preprocess_kwargs) # see documentation how it preprocesses the data

    def preprocessor_validation(PDFs):
        return pp.preprocess_2(PDFs, **preprocess_kwargs)

    # =========== dataset split ===========

    train_test_split_kwargs = {
        'val_size': 0.1,
        'test_size': 0.1,
        'random_state': 1234,
    }

    train_test_split = partial(tu.regression_train_test_split_4, **train_test_split_kwargs)


    def transform_X_fun(X):
        X = dg.random_detector_saturation_tf(X,
                                             saturation_level_min=args.saturation_level_min,
                                             saturation_level_max=args.saturation_level_max,
                                             )
        X = dg.random_flip_lrud_tf(X)
        X = dg.random_contrast_tf(X, contrast_min=0.1, contrast_max=1.0)
        X = dg.random_brightness_tf(X, brightness_max_delta=1.0)
        X = tf.clip_by_value(X, -1, 1)
        return X


    def transform_y_fun(y):
        return y


    data_gen_train = partial(dg.DataGenerator, transform_X_fun=transform_X_fun,
                             transform_y_fun=transform_y_fun,
                             batch_size=args.batch_size,
                             shuffle=True)
    data_gen_val = partial(dg.DataGenerator, transform_X_fun=None,
                           transform_y_fun=None,
                           batch_size=args.batch_size,
                           shuffle=True)
    data_gen_test = partial(dg.DataGenerator, transform_X_fun=None,
                            transform_y_fun=None,
                            batch_size=args.batch_size,
                            shuffle=True)

    # =========== training hyperparameters ============

    n_models = args.n_models

    model_funs = {
        args.model_name: partial(nets.deepCNN_pretrained_bayesian,
                          pretrained_model=getattr(tf.keras.applications, args.model_name) if 'ConvNeXt' not in args.model_name else getattr(tf.keras.applications.convnext, args.model_name),
                          dropout_rate=0.2,
                          weights='imagenet',
                          pooling='avg',
                          dense_layers=args.dense_layers,
                          )
    }

    lr_scheduler_kwargs = {
        'decay_rate': 0.5,
        'decay_step': 50
    }


    def callbacks(tensorboard_path):
        return [
            # PlotLossesKeras(),
            tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path,
                                           #histogram_freq=1,
                                           #profile_batch='50,70',
                                           write_images=True,
                                           #embeddings_freq=1,
                                           ),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100),
            tf.keras.callbacks.LearningRateScheduler(tu.lr_scheduler(**lr_scheduler_kwargs)),
            *[tu.ExtraValidation(*(ds_val.get_Xy()), tensorboard_path + '/extra_validation', ds_val.basename)
              for ds_val in ds_extra_val],
        ]


    def optimizer():
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        #tu.to_accumulative(opt, update_params_frequency=60)
        return opt


    def optimizer_fine_tune():
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        #tu.to_accumulative(opt, update_params_frequency=60)
        return opt


    def kernel_regularizer():
        #return tf.keras.regularizers.L1L2(l1=1e-7, l2=1e-7)
        return None


    def loss():
        return tu.RegressionNLL()


    training_kwargs = {
        'epochs': 50,
        'epochs_fine_tune': 150,
        'optimizer': optimizer,
        'loss': loss,
        'callbacks': callbacks,
        'callbacks_fine_tune': callbacks,
        'checkpoint': True,
        'checkpoint_filename': args.checkpoint_filename,
        'fine_tune': True,
        'optimizer_fine_tune': optimizer_fine_tune,
        'kernel_regularizer': kernel_regularizer,
    }

    # Possible bug to avoid: callbacks(?), optimizer and loss objects cannot
    # be shared between model instances as they will connect graphs.
    # Then ValueError: Unable to create dataset (name already exists),

    # ============= END CONFIGURATION ================

    ds = AttoDataset(dataset_path)
    ds.preprocess(preprocessor, feature)


    for ds_val in ds_extra_val:
        ds_val.preprocess(preprocessor_validation, feature)

    for model_number in range(args.random_state, args.random_state + n_models):
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(*ds.get_Xy(),
                                                                          random_state_validation=model_number)
        dg_train = data_gen_train(X_train, y_train)
        dg_val = data_gen_val(X_val, y_val)
        dg_test = data_gen_test(X_test, y_test)

        for model_name, model_fun in model_funs.items():
            training_kwargs['model_save_filename'] = f'{model_save_folder}/{ds.basename}_SL_{args.saturation_level_min}__Bayesian{model_name}__{model_number}.h5'
            training_kwargs['log_filename'] = model_save_folder + '/logs.txt'
            training_kwargs['callbacks'] = lambda: callbacks(training_kwargs['model_save_filename'][:-3])
            training_kwargs['callbacks_fine_tune'] = lambda: callbacks(
                tensorboard_path=training_kwargs['model_save_filename'][:-3] + '__fine_tune')

            model = tu.model_compile_train_save(dg_train, dg_val, dg_test,
                                                model_fun, **training_kwargs)
            print('GPU USAGE:')
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"current mem = {mem_info['current'] / 1073741824} GB")
            print(f"peak mem = {mem_info['peak'] / 1073741824} GB")
            tf.compat.v1.reset_default_graph()
            del model
            tf.keras.backend.clear_session()

        del X_train, X_test, X_val, y_train, y_val, y_test
