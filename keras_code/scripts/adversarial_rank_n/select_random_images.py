import os
import numpy as np
from scipy.io import loadmat
import pickle


result_folder = './data/'
image_folder = '/home/brais/Descargas/ILSVRC2012_img_val/'
label_filename = '/home/brais/Descargas/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
meta_filename = '/home/brais/Descargas/ILSVRC2012_devkit_t12/data/meta.mat'
images_per_label = 1
result_filename = 'imagenet_selected_images_' + str(images_per_label) + '.pickle'


def get_filenames(folder):
    return list(sorted([os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]))


if __name__ == '__main__':
    os.chdir('../../../')

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

    ulabels = np.unique(image_labels)

    selected_filenames = []
    selected_labels = []
    for label in ulabels:
        pos = np.where(image_labels == label)[0]
        label_positions = np.random.permutation(len(pos))[:images_per_label]
        for label_p in label_positions:
            p = pos[label_p]
            selected_filenames.append(image_filenames[p])
            selected_labels.append(label)

    selected_images = {'filenames': selected_filenames, 'labels': selected_labels}
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    with open(result_folder + result_filename, 'wb') as handle:
        pickle.dump(selected_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
