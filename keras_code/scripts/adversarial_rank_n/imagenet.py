import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_code.src import backend
from keras_code.scripts import networks, model_folder, info_folder
from keras_code.src.AdversarialRankN import AdversarialRankN
from keras_code.src.generators import FromDiskGenerator
from keras import backend as K, optimizers
from keras_code.src.utils import inverse_prepocess
import numpy as np
from keras.utils import to_categorical
import json
import pickle

# config = K.tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
#
# K.tensorflow_backend.set_session(K.tf.Session(config=config))

image_folder = '/home/brais/Descargas/ILSVRC2012_img_val/'
images_per_label = 1
data_folder = './data/'
data_filename = data_folder + 'imagenet_selected_images_' + str(images_per_label) + '.pickle'
filename = 'imagenet_rank.json'
min_max_filename = 'imagenet_min_max_input.json'
image_batch_size = 1000
batch_size = 15
alpha = 1e-4
Ns = [1]
optimizer = optimizers.Adam(1e-3)


def get_filenames(folder):
    return list(sorted([os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]))


def inverse_func(mode):

    def func(x):
        return inverse_prepocess(x, mode)
    return func


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
    image_labels = selected_images['labels']

    with open(info_folder + min_max_filename) as outfile:
        min_max_info = json.load(outfile)

    for network, preprocess, type in networks:

        network_name = network.__name__
        inverse_transform = inverse_func(type)

        model = network(include_top=True, weights='imagenet')
        model.trainable = False
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

        input_shape = model.input_shape[1:]
        nclasses = model.output_shape[1]

        image_generator = FromDiskGenerator(
            image_filenames, target_size=input_shape, batch_size=image_batch_size, preprocess_func=preprocess
        )

        y = to_categorical(image_labels, nclasses)

        print('NETWORK :', network_name)
        if network_name in min_max_info:
            print('using constraint : ', min_max_info[network_name])
            constrain_func = boundary_constraint(
                min_max_info[network_name]['min'], min_max_info[network_name]['max']
            )
        else:
            constrain_func = None

        with graph.as_default():
            adversarial_rank = AdversarialRankN(model=model)
            scores = adversarial_rank.get_adversarial_scores(
                image_generator, y, Ns=Ns, batch_size=batch_size, alpha=alpha, constraint=constrain_func, save_data=True,
                inverse_transform=inverse_transform
            )

        if not os.path.isdir(info_folder):
            os.makedirs(info_folder)

        try:
            with open(info_folder + filename) as outfile:
                info_data = json.load(outfile)
        except:
            info_data = {}

        if network_name not in info_data:
            info_data[network_name] = {}

        if 'untargeted' not in info_data[network_name]:
            info_data[network_name]['untargeted'] = {}

        for i in Ns:
            if i not in selected_images:
                selected_images[i] = {}
            selected_images[i]['adversarial_data'] = scores[i]['adversarial_data']
            del scores[i]['adversarial_data']
            info_data[network_name]['untargeted'][i] = scores[i]

        with open(info_folder + filename, 'w') as outfile:
            json.dump(info_data, outfile)

        network_folder = data_folder + network_name + '/'
        if not os.path.isdir(network_folder):
            os.makedirs(network_folder)
        adv_filename = network_folder + 'adversarial_data_try_' + str(images_per_label) + '.pickle'
        with open(adv_filename, 'wb') as handle:
            pickle.dump(selected_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # del model
        # K.clear_session()