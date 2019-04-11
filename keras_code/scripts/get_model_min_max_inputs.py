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
filename = 'imagenet_rank.json'
min_max_filename = 'imagenet_min_max_input.json'
image_batch_size = 1000
batch_size = 15
alpha = 1e-4
Ns = [5, 1]
optimizer = optimizers.Adam(1e-3)


def get_filenames(folder):
    return list(sorted([os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]))


if __name__ == '__main__':
    os.chdir('../../')
    min_max_info = {}

    with open(info_folder + filename) as outfile:
        info_data = json.load(outfile)

    for network, preprocess in networks:

        network_name = network.__name__

        min_max_info[network_name] = {
            'min': info_data[network_name]['min'], 'max': info_data[network_name]['max']
        }

    with open(info_folder + min_max_filename, 'w') as outfile:
        json.dump(min_max_info, outfile)
