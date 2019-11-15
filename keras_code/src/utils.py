import numpy as np
from keras.preprocessing import image


def inverse_prepocess(x, mode='torch'):

    if mode == 'torch':
        mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,1,3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1,1,1,3))
        x = 255. * (x * std + mean)
    elif mode == 'tf':
        x = (x + 1) * 127.5
    else:
        mean = np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))
        x += mean

    return x


def load_images(filenames, target_size):
    x = np.array(
        [image.img_to_array(image.load_img(filename, target_size=target_size[:2], interpolation='bicubic')) for
        filename in filenames]
    )
    return x