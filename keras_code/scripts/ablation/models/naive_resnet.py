from keras import layers, optimizers, models, losses, regularizers, backend as K


class NaiveResNet:

    def naive_residual(self, inputs, bn=True, strides=(1, 1), k=1):
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = inputs
        if bn:
            x = layers.BatchNormalization(axis=bn_axis, scale=False)(x)  # Specifying the axis and mode allows for later merging
        x = layers.Activation('relu')(x)

        wire = layers.Convolution2D(16 * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
            use_bias=False, kernel_regularizer=regularizers.l2(5e-4))(x)

        x = layers.Convolution2D(16 * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
            use_bias=False, kernel_regularizer=regularizers.l2(5e-4))(x)
        if bn:
            x = layers.BatchNormalization(axis=bn_axis, scale=False)(x)  # Specifying the axis and mode allows for later merging
        x = layers.Activation('relu')(x)
        x = layers.Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
            use_bias=False, kernel_regularizer=regularizers.l2(5e-4))(x)

        x = layers.Add()([wire, x])

        return x

    def __init__(self, shape, nclasses, bn=True, gap=False, k=1):
        inputs = layers.Input(shape=shape)

        x = layers.Convolution2D(
            16, (3, 3), padding='same', kernel_initializer='he_normal',
            use_bias=False, kernel_regularizer=regularizers.l2(5e-4)
        )(inputs)
        x = self.naive_residual(x, bn=bn, strides=(2, 2), k=k)
        x = self.naive_residual(x, bn=bn, strides=(2, 2), k=k)
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
        x = layers.Activation('relu')(x)

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
