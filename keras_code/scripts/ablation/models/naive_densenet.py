from keras import layers, optimizers, models, losses, regularizers, backend as K


class NaiveDenseNet:

    def conv_block(self, inputs, bn=True, k=1):
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = inputs
        if bn:
            x = layers.BatchNormalization(axis=bn_axis, scale=False)(x)  # Specifying the axis and mode allows for later merging
        x = layers.Activation('relu')(x)

        x = layers.Convolution2D(16 * k, (1, 1), padding='same', kernel_initializer='he_normal',
            use_bias=False, kernel_regularizer=regularizers.l2(5e-4))(x)
        if bn:
            x = layers.BatchNormalization(axis=bn_axis, scale=False)(x)  # Specifying the axis and mode allows for later merging
        x = layers.Activation('relu')(x)
        x = layers.Convolution2D(4 * k, (3, 3), padding='same', kernel_initializer='he_normal',
            use_bias=False, kernel_regularizer=regularizers.l2(5e-4))(x)

        x = layers.Concatenate(axis=bn_axis)([inputs, x])

        return x

    def transition_block(self, x, reduction):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                          use_bias=False)(x)
        x = layers.AveragePooling2D(2, strides=2)(x)
        return x


    def naive_dense(self, x, blocks, bn=True, k=1):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, bn=bn, k=k)
        x = self.transition_block(x, 0.5)
        return x


    def __init__(self, shape, nclasses, bn=True, gap=False, k=1):
        inputs = layers.Input(shape=shape)

        x = layers.Convolution2D(
            16, (3, 3), padding='same', kernel_initializer='he_normal',
            use_bias=False, kernel_regularizer=regularizers.l2(5e-4)
        )(inputs)
        x = self.naive_dense(x, 4, bn=bn, k=k)
        x = self.naive_dense(x, 4, bn=bn, k=k)
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
                      optimizer=optimizers.SGD(lr=1e-2 if bn else 1e-4),
                      metrics=['accuracy'])
        self.model = model
        self.shape = shape
        self.nclasses = nclasses
        self.bn = bn
        self.gap = gap
        self.k = k
