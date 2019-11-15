from keras_code.src import backend
import os
from keras_code.scripts import networks, model_folder, info_folder
from keras_code.src.AdversarialRankN import AdversarialRankN
from keras_code.src.generators import FromDiskGenerator
from keras_code.src.utils import inverse_prepocess, load_images
from keras import backend as K, optimizers
import numpy as np
from keras.utils import to_categorical
import json
import pickle
from scipy.misc import imresize


config = K.tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

K.tensorflow_backend.set_session(K.tf.Session(config=config))

image_folder = '/home/brais/Descargas/ILSVRC2012_img_val/'
images_per_label = 1
data_folder = './data/'
data_folder = '/media/brais/Data/data/'
data_filename = data_folder + 'imagenet_selected_images_' + str(images_per_label) + '.pickle'
filename = 'imagenet_rank.json'
min_max_filename = 'imagenet_min_max_input.json'
image_batch_size = 1000
batch_size = 25
alpha = 1e-4
Ns = 1
optimizer = optimizers.Adam(1e-3)


def get_filenames(folder):
    return list(sorted([os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]))


def boundary_constraint(minval, maxval):

    def func(x):
        x[x < minval] = minval
        x[x > maxval] = maxval
        return x

    return func


if __name__ == '__main__':
    os.chdir('../../../')

    graph = K.tf.get_default_graph()

    with open(data_filename, 'rb') as handle:
        selected_images = pickle.load(handle)
    image_filenames = selected_images['filenames']
    image_labels = np.array(selected_images['labels']).reshape((-1, 1))

    with open(info_folder + min_max_filename) as outfile:
        min_max_info = json.load(outfile)

    for network_2, preprocess, _ in networks:

        model = network_2(include_top=True, weights='imagenet')
        model.trainable = False
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        network_2_name = network_2.__name__

        network_folder = data_folder + network_2_name + '/'
        adv_filename = network_folder + 'adversarial_data_phishing_' + str(Ns) + '_095_' + str(
            images_per_label) + '.pickle'
        with open(adv_filename, 'rb') as handle:
            unpickler = pickle.Unpickler(handle)
            info = unpickler.load()
        images = load_images(info['filenames'], model.input_shape[1:])
        images = preprocess(images)
        y_pred = model.predict(images, batch_size=batch_size, verbose=0)
        y_pred_argsort = np.argsort(-1. * y_pred, axis=-1)
        y_targets = y_pred_argsort[y_pred_argsort != image_labels].reshape((y_pred_argsort.shape[0], -1))[:, (Ns - 1):Ns]
        del info

        for network, _, mode in networks:
            network_name = network.__name__

            network_folder = data_folder + network_name + '/'
            adv_filename = network_folder + 'adversarial_data_phishing_' + str(Ns) + '_095_' + str(images_per_label) + '.pickle'
            with open(adv_filename, 'rb') as handle:
                unpickler = pickle.Unpickler(handle)
                info = unpickler.load()
            adv_images = info[str(Ns) + '_0.95']['adversarial_data']
            if adv_images.shape[1:] != model.input_shape[1:]:
                adv_images = np.array([imresize(x, model.input_shape[1:], 'bicubic') for x in adv_images])
            adv_images = inverse_prepocess(adv_images, mode=mode)
            adv_images = preprocess(adv_images)

            predicted_output = model.predict(adv_images, batch_size=batch_size, verbose=0)
            predicted_label = np.argsort(-1. * predicted_output, axis=-1)[:, :1]
            predicted_score = -1 * np.sort(-1. * predicted_output, axis=-1)[:, :1]
            error_label = np.mean(predicted_label == y_targets)
            error_label_score = np.mean((predicted_label == y_targets) & (predicted_score > 0.9))
            error_score = np.mean((predicted_label != image_labels) & (predicted_score > 0.9))
            print('NETWORK ADV IMAGES : ', network_2_name, ', NETWORK : ', network_name, ', LABEL : ', error_label, ', LABEL_PREC : ', error_label_score, ', PREC : ', error_score)
            del info

        del model
        K.clear_session()
