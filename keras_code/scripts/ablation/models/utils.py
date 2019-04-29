from keras import layers, backend as K, regularizers


def conv2d_bn(filters,
              kernel_size,
              padding='same',
              strides=(1, 1),
              activation='relu',
              bn=True,
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    def func(x):

        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            use_bias=not bn,
            name=conv_name,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(5e-4)
        )(x)
        if bn:
            x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = layers.Activation(activation, name=name)(x)
        return x

    return func
