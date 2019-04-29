from keras import layers, optimizers, models, losses, regularizers
from keras_code.scripts.ablation.models.utils import conv2d_bn


def naive_inception(inputs, bn=True, strides=(1, 1), k=1):

    towerOne = conv2d_bn(6*k, (1,1), activation='relu', padding='same', bn=bn, strides=strides)(inputs)
    towerTwo = conv2d_bn(6*k, (3,3), activation='relu', padding='same', bn=bn, strides=strides)(inputs)
    towerThree = conv2d_bn(6*k, (5,5), activation='relu', padding='same', bn=bn, strides=strides)(inputs)
    x = layers.concatenate([towerOne, towerTwo, towerThree], axis=3)
    return x


def dimension_reduction_inception(inputs, bn=True, strides=(1, 1), k=1):
    tower_one = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_one = conv2d_bn(6*k, (1,1), activation='relu', padding='same', bn=bn, strides=strides)(tower_one)

    tower_two = conv2d_bn(6*k, (1,1), activation='relu', padding='same', bn=bn)(inputs)
    tower_two = conv2d_bn(6*k, (3,3), activation='relu', padding='same', bn=bn, strides=strides)(tower_two)

    tower_three = conv2d_bn(6*k, (1,1), activation='relu', padding='same', bn=bn)(inputs)
    tower_three = conv2d_bn(6*k, (5,5), activation='relu', padding='same', bn=bn, strides=strides)(tower_three)
    x = layers.concatenate([tower_one, tower_two, tower_three], axis=3)
    return x


def naive_model(shape, nclasses, bn=True, gap=False, k=1, dimension_reduction=False):

    inputs = layers.Input(shape=shape)

    inception_func = dimension_reduction_inception if dimension_reduction else naive_inception

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
    return model
