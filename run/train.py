import sys
import os
from functools import partial

import tensorflow as tf

from attoDNN.attodataset import AttoDataset
from attoDNN import nets, train_utils as tu, preprocess as pp, data_generator as dg

if __name__ == "__main__":
    print(nets.list_GPUs())

    #tu.set_memory_growth()

    # ============= CONFIGURATION ================
    model_save_folder = '/nfs/amd0/home/c8888/attosecond_ML/models/attoDNN/preprocess_2_remove_origin'
    datasets_paths = [
                      '/nfs/amd0/home/c8888/attosecond_ML/data_cleaned/cartesian_0/QProp/QProp_Ar_4590_LongPulse_10000_joined_intensity_avg_relative_uncertainty_0.10.npz',
                     ]

    # =========== preprocessing ===========

    feature = 'Up'

    preprocess_kwargs = {
        'threshold': 1e-6,
        'downsample_1': 1,
        'downsample_2': 1,
    }
    preprocessor = lambda PDFs: pp.remove_region_around_origin(pp.preprocess_2(PDFs, **preprocess_kwargs),
                                                               removal_size=(0.7, 0.4))


    # =========== dataset split ===========

    train_test_split_kwargs = {
        'val_size': 0.1,
        'test_size': 0.1,
        'random_state': 1234,
    }


    train_test_split = partial(tu.regression_train_test_split_2, **train_test_split_kwargs)

    def transform_X_fun(X):
        X = dg.random_background_noise_for_preprocess_2_tf(X,
                                                          preprocess_kwargs['threshold'],
                                                          noise_level=1e-4)
        X = dg.random_flip_lrud_tf(X)
        X = dg.random_contrast_tf(X, contrast_min=0.2, contrast_max=1.0)
        X = dg.random_brightness_tf(X, brightness_max_delta=0.2)
        X = tf.clip_by_value(X, -1, 1)
        return X

    def transform_y_fun(y):
        return y


    data_gen_train = partial(dg.DataGenerator, transform_X_fun=transform_X_fun,
                       transform_y_fun=transform_y_fun,
                       batch_size=10,
                       shuffle=True)
    data_gen_val = partial(dg.DataGenerator, transform_X_fun=None,
                       transform_y_fun=None,
                       batch_size=10,
                       shuffle=True)
    data_gen_test = partial(dg.DataGenerator, transform_X_fun=None,
                       transform_y_fun=None,
                       batch_size=10,
                       shuffle=True)


    # =========== training hyperparameters ============

    n_models = 5

    model_funs = {'SimpleCNN': nets.deepCNN1,
                  'Xception': partial(nets.deepCNN_pretrained,
                                      pretrained_model=tf.keras.applications.Xception,
                                      dropout_rate=0.2,
                                      weights='imagenet'),
                  'VGG16': partial(nets.deepCNN_pretrained,
                                   pretrained_model=tf.keras.applications.VGG16,
                                   dropout_rate=0.2,
                                   weights='imagenet'),
                  'ResNet50V2': partial(nets.deepCNN_pretrained,
                                        pretrained_model=tf.keras.applications.ResNet50V2,
                                        dropout_rate=0.2,
                                        weights='imagenet'),
                 }



    lr_scheduler_kwargs = {
        'decay_rate': 0.5,
        'decay_step': 50
    }


    def callbacks():
        return [#PlotLossesKeras(),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
                 tf.keras.callbacks.LearningRateScheduler(tu.lr_scheduler(**lr_scheduler_kwargs))
                ]


    def optimizer():
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        tu.to_accumulative(opt, update_params_frequency=30)
        return opt


    def optimizer_fine_tune():
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        tu.to_accumulative(opt, update_params_frequency=30)
        return opt


    def kernel_regularizer():
        return tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-5)


    def loss():
        return tf.keras.losses.MeanAbsoluteError()

    training_kwargs = {
        'epochs': 150,
        'optimizer': optimizer,
        'loss': loss,
        'callbacks': callbacks,
        'checkpoint': True,
        'checkpoint_filename': '/home/c8888/scratch/checkpoint/checkpoint',
        'fine_tune': True,
        'optimizer_fine_tune': optimizer_fine_tune,
        'kernel_regularizer': kernel_regularizer
    }

    # Possible bug to avoid: callbacks(?), optimizer and loss objects cannot
    # be shared between model instances as they will connect graphs.
    # Then ValueError: Unable to create dataset (name already exists),


    # ============= END CONFIGURATION ================

    #os.popen(f'cat {__file__} >> {model_save_folder}/{os.path.basename(__file__)}')  # save training settings

    datasets = [AttoDataset(fn) for fn in datasets_paths]

    for model_number in range(n_models):
        for ds in datasets:
            ds.preprocess(preprocessor, feature)

            X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(*ds.get_Xy())

            dg_train = data_gen_train(X_train, y_train)
            dg_val = data_gen_val(X_val, y_val)
            dg_test = data_gen_test(X_test, y_test)

            for model_name, model_fun in model_funs.items():
                training_kwargs['model_save_filename'] = f'{model_save_folder}/{ds.basename}__{model_name}__{model_number}.h5'
                training_kwargs['log_filename'] = model_save_folder + '/logs.txt'
                model = tu.model_compile_train_save(dg_train, dg_val, dg_test,
                                                    model_fun, **training_kwargs)
                tf.compat.v1.reset_default_graph()
                del model
                tf.keras.backend.clear_session()
