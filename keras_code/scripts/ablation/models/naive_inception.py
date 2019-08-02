from keras import layers, optimizers, models, losses, regularizers


class NaiveInception:

    def conv2d_bn(self, filters,
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

    def naive_inception(self, inputs, bn=True, strides=(1, 1), k=1):

        towerOne = self.conv2d_bn(6*k, (1,1), activation='relu', padding='same', bn=bn, strides=strides)(inputs)
        towerTwo = self.conv2d_bn(6*k, (3,3), activation='relu', padding='same', bn=bn, strides=strides)(inputs)
        towerThree = self.conv2d_bn(6*k, (5,5), activation='relu', padding='same', bn=bn, strides=strides)(inputs)
        x = layers.concatenate([towerOne, towerTwo, towerThree], axis=3)
        return x


    def dimension_reduction_inception(self, inputs, bn=True, strides=(1, 1), k=1):
        tower_one = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
        tower_one = self.conv2d_bn(6*k, (1,1), activation='relu', padding='same', bn=bn, strides=strides)(tower_one)

        tower_two = self.conv2d_bn(6*k, (1,1), activation='relu', padding='same', bn=bn)(inputs)
        tower_two = self.conv2d_bn(6*k, (3,3), activation='relu', padding='same', bn=bn, strides=strides)(tower_two)

        tower_three = self.conv2d_bn(6*k, (1,1), activation='relu', padding='same', bn=bn)(inputs)
        tower_three = self.conv2d_bn(6*k, (5,5), activation='relu', padding='same', bn=bn, strides=strides)(tower_three)
        x = layers.concatenate([tower_one, tower_two, tower_three], axis=3)
        return x


    def __init__(self, shape, nclasses, bn=True, gap=False, k=1, dimension_reduction=False):

        inputs = layers.Input(shape=shape)

        inception_func = self.dimension_reduction_inception if dimension_reduction else self.naive_inception

        x = layers.Convolution2D(
            16, (3, 3), padding='same', kernel_initializer='he_normal',
            use_bias=False, kernel_regularizer=regularizers.l2(5e-4)
        )(inputs)
        x = inception_func(x, bn=bn, strides=(2, 2), k=k)
        x = inception_func(x, bn=bn, strides=(2, 2), k=k)

        if gap:
            x = layers.Conv2D(nclasses, (3, 3), padding='VALID')(x)
            x = layers.GlobalAveragePooling2D()(x)
            predictions = layers.Flatten()(x)
        else:
            x = layers.Flatten()(x)
            predictions = layers.Dense(nclasses, activation='softmax')(x)

        model = models.Model(input=inputs, output=predictions)

        model.compile(loss=losses.categorical_crossentropy,
                     optimizer=optimizers.adam(lr=1e-2 if bn else 1e-4),
                     metrics=['accuracy'])
        self.model = model
        self.shape = shape
        self.nclasses = nclasses
        self.bn = bn
        self.gap = gap
        self.k = k
