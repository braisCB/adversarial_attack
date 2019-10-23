import os
import numpy as np
import pickle


images_per_label = 1
data_folder = './data/'
data_filename = data_folder + 'imagenet_selected_images_' + str(images_per_label) + '.pickle'


if __name__ == '__main__':
    os.chdir('../../../')

    with open(data_filename, 'rb') as handle:
        selected_images = pickle.load(handle)
    image_labels = selected_images['labels']

    adversarial_labels = np.random.randint(1000, size=len(image_labels))
    while np.any(image_labels == adversarial_labels):
        p = np.where(image_labels == adversarial_labels)[0]
        adversarial_labels[p] = np.random.randint(1000, size=len(p))

        adv_filename = data_folder + 'adversarial_labels_' + str(images_per_label) + '.pickle'
        with open(adv_filename, 'wb') as handle:
            pickle.dump(adversarial_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
