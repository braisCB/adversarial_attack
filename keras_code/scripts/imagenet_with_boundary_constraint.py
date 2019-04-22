from keras_code.src import backend
import os
from keras_code.scripts import networks, model_folder, info_folder
from keras_code.src.AdversarialRankN import AdversarialRankN
from keras_code.src.generators import FromDiskGenerator
from keras import backend as K, optimizers
import numpy as np
from keras.utils import to_categorical
import json
from scipy.io import loadmat


image_folder = '/home/brais/Descargas/ILSVRC2012_img_val/'
label_filename = '/home/brais/Descargas/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
meta_filename = '/home/brais/Descargas/ILSVRC2012_devkit_t12/data/meta.mat'
filename = 'imagenet_rank_with_constraint.json'
min_max_filename = 'imagenet_min_max_input.json'
image_batch_size = 1000
batch_size = 16
alpha = 1e-4
Ns = [5, 1]
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
    os.chdir('../../')

    graph = K.tf.get_default_graph()

    meta = loadmat(meta_filename)

    image_filenames = get_filenames(image_folder)
    synsets_index = np.zeros((1000,), dtype=int)
    synsets_names = []
    for i in range(1000):
        synsets_index[i] = int(meta['synsets'][i,0][0][0][0]) - 1
        synsets_names.append(meta['synsets'][i,0][1][0])

    # synsets_names = np.array(synsets_names)[synsets_index]
    order = np.argsort(synsets_names) + 1
    conversion = dict(zip(order, synsets_index))
    image_labels = np.loadtxt(label_filename, dtype=int)
    image_labels = np.array([conversion[x] for x in image_labels])

    with open(info_folder + min_max_filename) as outfile:
        min_max_info = json.load(outfile)

    for network, preprocess in networks:

        network_name = network.__name__

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
                image_generator, y, Ns=Ns, batch_size=batch_size, alpha=alpha, constraint=constrain_func
            )

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
