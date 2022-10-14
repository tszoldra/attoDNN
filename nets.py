from tensorflow import keras


def deepCNN_pretrained(input_shape, output_dim=1,
                       pretrained_model=keras.applications.Xception,
                       dropout_rate=0., weights='imagenet'):
    """
    Model with a pretrained part with the last layer removed
    and a GlobalAveragePooling2D, Dropout and a single linear dense layer applied.

    Arguments:
    :param input_shape: Tuple with ``(image_width, image_height, n_channels)``.
    :param output_dim: Dimensionality of the output. For prediction of a scalar, set to 1.
    :param pretrained_model: One of the ``tf.keras.applications`` classes.
    :param dropout_rate: Float between 0 and 1. The Dropout layer randomly sets input units to 0
        with a frequency of rate at each step during training time, which helps prevent overfitting.
    :param weights: Either ``imagenet`` for a model pretrained on imagenet dataset or None for \
        random initialization. For random initialization, the base_model becomes trainable.

    :returns: (model, base_model) tuple where ``model`` is the whole model and `base_model` is the pretrained part.
        Model needs to be compiled.
    """

    base_model = pretrained_model(include_top=False,
                                  input_shape=input_shape,
                                  weights=weights)

    if weights is not None:
        base_model.trainable = False
    elif weights == 'imagenet':
        base_model.trainable = True
    else:
        raise ValueError('Weights must be either "imagenet" or None.')

    x = base_model.output
    # x = base_model(x, training=False)  # TODO Important for batchNormalization layer; see https://keras.io/guides/transfer_learning/
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(output_dim, activation='linear')(x)

    model = keras.models.Model(base_model.inputs, outputs)

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
