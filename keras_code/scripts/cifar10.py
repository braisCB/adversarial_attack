from keras_code.src.AdversarialRankN import AdversarialRankN
from keras_code.scripts import small_networks, model_folder, info_folder
from keras.datasets import cifar10
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, regularizers, optimizers, initializers
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import os
import json
from keras.utils import to_categorical


finetuning = False
optimizer = optimizers.adam(lr=1e-3)
weights = None
adam=True
pooling = 'avg'
epochs = 40 if finetuning else 130
batch_size = 128
Ns = [1]
force_training = False
alpha = 5e-5

filename = 'cifar10_finetuning_' + str(finetuning) + '_adam_' + str(adam) + '_weights_' + str(weights) +'.json'
print(filename)

generator = ImageDataGenerator(
    width_shift_range=5. / 32,
    height_shift_range=5. / 32,
    fill_mode='reflect',
    horizontal_flip=True
)


def scheduler(adam=True):
    def sch(epoch):
        if epoch < 40:
            lr = .1
        elif epoch < 80:
            lr = .02
        elif epoch < 115:
            lr = .004
        else:
            lr = .0008
        if adam:
            lr /= 10.
        print('lr: ', lr)
        return lr

    return sch


def reset_weights(model, kernel_initializer='he_normal'):
    initial_weights = model.get_weights()
    k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
    new_weights = [k_eval(initializers.get(kernel_initializer)(w.shape)) for w in initial_weights]
    model.set_weights(new_weights)


if __name__ == '__main__':
    os.chdir('../../')

    # x_train = x_train / 127.5 - 1.
    # x_test = x_test / 127.5 - 1.

    for network, preprocess in small_networks:

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # x_train = x_train / 127.5 - 1.
        # x_test = x_test / 127.5 - 1.

        x_train = preprocess(x_train)
        x_test = preprocess(x_test)

        print(x_train.max(), x_train.min())

        input_shape = (32, 32, 3)
        classes = len(np.unique(y_train))

        y_train = to_categorical(y_train, classes)
        y_test = to_categorical(y_test, classes)

        network_name = network.__name__
        network_file = model_folder + 'cifar10_' + network_name + '_finetuning_' + str(finetuning) + \
                       '_adam_' + str(adam) + '_weights_' + str(weights) + '.h5'

        if os.path.isfile(network_file) and not force_training:
            model = models.load_model(network_file)
        else:
            base_model = network(include_top=True, weights=weights, input_shape=input_shape, pooling=None, classes=classes)

            if weights is None:
                reset_weights(base_model)

            if finetuning:
                base_model.trainable = False
            else:
                base_model.trainable = True

            x = base_model(base_model.input)

            # x = layers.Flatten()(x)
            # #
            # x = layers.Dense(1024, use_bias=False, kernel_initializer='he_normal',
            #                  kernel_regularizer=regularizers.l2(5e-4))(x)
            # x = layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
            # x = layers.Activation('relu')(x)
            #
            # x = layers.Dense(classes, use_bias=True, kernel_initializer='he_normal',
            #                  kernel_regularizer=regularizers.l2(5e-4))(x)
            # x = layers.Activation('softmax')(x)

            # this is the model we will train
            model = models.Model(inputs=base_model.inputs, outputs=x)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

            model.summary()

            callbacks = []
            if not finetuning:
                callbacks.append(
                    LearningRateScheduler(scheduler(adam))
                )

            model.fit_generator(
                generator.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs,
                callbacks=callbacks,
                validation_data=(x_test, y_test),
                validation_steps=x_test.shape[0] // batch_size,
                verbose=2
            )

            if not os.path.isdir(model_folder):
                os.makedirs(model_folder)

            model.save(filepath=network_file)

        adversarial_rank = AdversarialRankN(model=model)
        scores = adversarial_rank.get_adversarial_scores(
            x_test, y_test, Ns=Ns, batch_size=batch_size, alpha=alpha
        )

        del adversarial_rank
        del model
        K.clear_session()

        if not os.path.isdir(info_folder):
            os.makedirs(info_folder)

        try:
            with open(info_folder + filename) as outfile:
                info_data = json.load(outfile)
        except:
            info_data = {}

        info_data[network_name] = scores

        with open(info_folder + filename, 'w') as outfile:
            json.dump(info_data, outfile)
