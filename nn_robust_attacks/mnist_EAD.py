import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_code.src import backend
from keras_code.src.AdversarialPhishingN import AdversarialPhishingN
from keras import backend as K, optimizers
import numpy as np
from setup_mnist import MNIST, MNISTModel
from keras.utils import to_categorical
import json
import pickle

config = K.tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99

K.tensorflow_backend.set_session(K.tf.Session(config=config))

image_folder = '/home/brais/Descargas/ILSVRC2012_img_val/'
images_per_label = 1
data_folder = './data/'
data_filename = data_folder + 'imagenet_selected_images_' + str(images_per_label) + '.pickle'
filename = 'imagenet_rank.json'
min_max_filename = 'imagenet_min_max_input.json'
image_batch_size = 1000
batch_size = 100
alpha = 1e-2
threshs = [0.]
optimizer = optimizers.Adam(1e-3)

n = 100


def boundary_constraint(minval=-0.5, maxval=0.5):

    def func(x):
        x[x < minval] = minval
        x[x > maxval] = maxval
        return x

    return func


def inverse_func(x):
    return (x + 0.5) * 255


if __name__ == '__main__':
    path = os.getcwd()
    os.chdir('../../../')

    # avd_filename = data_folder + 'adversarial_labels_' + str(images_per_label) + '.pickle'
    # with open(avd_filename, 'rb') as handle:
    #     n = to_categorical(pickle.load(handle), 1000)

    with K.tf.Session() as sess:
        dataset, model =  MNIST(), MNISTModel(path + '/models/mnist-distilled-80', sess, True).model
        model.trainable = False
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        len_test = len(dataset.test_data)

        p = np.arange(len_test, dtype=int)
        np.random.shuffle(p)
        p = p[:1000]
        data = dataset.test_data[p]
        label = dataset.test_labels[p]
        y_label = np.argmax(label, axis=1)

        n = np.random.randint(10, size=(1000,))
        while np.any(n == y_label):
            p = np.where(n == y_label)[0]
            n[p] = np.random.randint(10, size=(len(p),))

        constrain_func = boundary_constraint(-0.5, 0.5)

        adversarial_rank = AdversarialPhishingN(model=model)
        scores = adversarial_rank.get_adversarial_scores(
            data, label, n=to_categorical(n, 10), threshs=threshs, batch_size=batch_size, alpha=alpha, constraint=constrain_func,
            save_data=True, inverse_transform=inverse_func
        )


