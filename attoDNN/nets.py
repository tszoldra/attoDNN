from tensorflow import keras
import tensorflow as tf


def list_GPUs():
    return tf.config.list_physical_devices('GPU')


def deepCNN_pretrained(input_shape, output_dim=1,
                       pretrained_model=keras.applications.Xception,
                       dropout_rate=0., weights='imagenet',
                       pooling='avg',
                       dense_layers=None):
    """
    Model with a pretrained part, pooling, Dropout and dense layers.

    Arguments:
    :param input_shape: Tuple with ``(image_width, image_height, n_channels)``. If n_channels=1,
        channels are repeated 3 times.
    :param output_dim: Dimensionality of the output. For prediction of a scalar, set to 1.
    :param pretrained_model: One of the ``tf.keras.applications`` classes.
    :param dropout_rate: Float between 0 and 1. The Dropout layer randomly sets input units to 0
        with a frequency of rate at each step during training time, which helps prevent overfitting.
    :param weights: Either ``imagenet`` for a model pretrained on imagenet dataset or None for \
        random initialization. For random initialization, the base_model becomes trainable.
    :param pooling: pooling applied after the pretrained part. Can be 'avg', 'max' or None.
    :param dense_layers: Either None for a single output layer or list [n1, n2,...] where \
        n1, n2, ... are dense layers dimensions.


    :returns: (model, base_model) tuple where ``model`` is the whole model and `base_model` is the pretrained part.
        Model needs to be compiled.
    """
    inputs = keras.Input(shape=input_shape)

    if input_shape[2] == 1:
        inputs = tf.repeat(inputs, 3, -1)

    if 'EfficientNetB' in str(pretrained_model):
        inputs = tf.keras.layers.Rescaling(scale=127.5, offset=1)(inputs)  # inefficient workaround for
                                                                   # EfficientNet V1 models
    pretrained_model_kwargs = dict(input_tensor=inputs,
                                  include_top=False,
                                  weights=weights,
                                  pooling=pooling)
    if ('EfficientNetV2' in str(pretrained_model)) or ('ConvNeXt' in str(pretrained_model)):
        pretrained_model_kwargs['include_preprocessing'] = False

    base_model = pretrained_model(**pretrained_model_kwargs)

    if weights is None:
        base_model.trainable = True
    elif weights == 'imagenet':
        base_model.trainable = False
    else:
        raise ValueError('Weights must be either "imagenet" or None.')

    # un-freeze the BatchNorm layers
    # https://datascience.stackexchange.com/questions/47966/over-fitting-in-transfer-learning-with-small-dataset
    for layer in base_model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True
    # x = base_model(x, training=False)  # Important for batchNormalization layer; here not desired.
                                         # see https://keras.io/guides/transfer_learning/
    x = base_model.output  # workaround for innvestigate - model cannot have one complicated layer but many simple layers!
    x = keras.layers.Dropout(dropout_rate)(x)
    if dense_layers is None:
        outputs = keras.layers.Dense(output_dim, activation='linear')(x)
    else:
        outputs = keras.layers.Dense(dense_layers[0], activation='relu')(x)
        for n in dense_layers[1:]:
            outputs = keras.layers.Dense(n, activation='relu')(outputs)
        outputs = keras.layers.Dense(output_dim, activation='linear')(outputs)

    model = keras.models.Model(base_model.input, outputs)

    return model, base_model



def deepCNN1(input_shape, output_dim):
    """
    Simple model with 3 convolutional layers.

    Arguments:
    :param input_shape: Tuple with ``(image_width, image_height, n_channels)``.
    :param output_dim: Dimensionality of the output. For prediction of a scalar, set to 1.

    :returns: (model, []) tuple where ``model`` is the whole model that needs to be compiled.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(output_dim))

    return model, []
