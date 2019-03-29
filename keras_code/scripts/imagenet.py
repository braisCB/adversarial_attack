from keras_code.src import backend
import os
from keras_code.scripts import networks, model_folder, info_folder
from keras_code.src.AdversarialRankN import AdversarialRankN
from keras_code.src.generators import FromDiskGenerator
from keras import backend as K
import numpy as np
from keras.utils import to_categorical
import json


image_folder = 'path_to_images'
label_filename = 'label_filename'
filename = 'imagenet.json'
image_batch_size = 1000
batch_size = 15
alpha = 1e-4
Ns = [1]


def get_filenames(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.join(folder, f))]


if __name__ == '__main__':
    os.chdir('../../')

    graph = K.tf.get_default_graph()

    image_filenames = get_filenames(image_folder)
    image_labels = np.loadtxt(label_filename)

    for network, preprocess in networks:

        model = network(include_top=True, weights='imagenet')

        input_shape = model.input_shape[1:]
        nclasses = model.output_shape[1]

        image_generator = FromDiskGenerator(
            image_filenames, target_size=input_shape, batch_size=batch_size, preprocess_func=preprocess
        )

        y = to_categorical(image_labels, nclasses)

        network_name = network.__name__

        with graph.as_default():
            adversarial_rank = AdversarialRankN(model=model)
            scores = adversarial_rank.get_adversarial_scores(
                image_generator, y, batch_size=batch_size, alpha=alpha
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
