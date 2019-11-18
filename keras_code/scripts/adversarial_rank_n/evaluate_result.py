import os
from keras_code.scripts import networks
from keras_code.src.generators import FromDiskGenerator
from keras_code.src.utils import inverse_prepocess
import pickle
import numpy as np
from matplotlib import pyplot as plt


image_folder = '/home/brais/Descargas/ILSVRC2012_img_val/'
images_per_label = 1
data_folder = './data/'
data_filename = data_folder + 'imagenet_selected_images_' + str(images_per_label) + '.pickle'
filename = 'imagenet_rank.json'
image_batch_size = 1000
batch_size = 15
alpha = 1e-4
Ns = [1]


def getsize(network_name):
    if network_name == 'NASNetLarge':
        return (331,331,3)
    if network_name in ['InceptionResNetV2', 'InceptionV3', 'Xception']:
        return (299,299,3)
    else:
        return (224,224,3)


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


L2_1 = np.array([0.10481569584287277, 0.11732080131206406, 0.031102971807694942, 0.03265520859221519, 0.038851468863943805, 0.14825613175145588,
        0.08276697675237035, 0.11030256937177461, 0.11964815122467644, 0.12542981119281643, 0.034627245993879785, 0.0367448642590198,
        0.2596189191423969, 0.29330659515729046])
L1_1 = np.array([34.96459341129815, 38.84181743945116, 23.378246685936837, 25.772921759041086, 31.22091243456446, 51.11713189012778,
        23.809900043476954, 35.460316464686066, 41.21469526254459, 43.96651747208318, 27.965674640969294, 29.862983110759444,
        106.26290961879526, 121.56799556005073])
L0_1 = np.array([19399.832, 23941.508, 18750.544, 21075.391, 22498.813, 21947.489,
        16163.425, 20466.334, 21275.905, 22558.375, 23324.158, 24067.234,
        28999.921, 35884.1])
Linf_1 = np.array([0.0010888780247351386, 0.001405850220006527, 0.000782983388546201, 0.0008097900764445391, 0.0009614110386975233, 0.0021947088167046038,
          0.0009353950677099574, 0.0016061715915500659, 0.0013016411985994308, 0.001406026923939772, 0.0008831191136122637, 0.0009426527894535239,
          0.00477120686465582, 0.00582289496828648])
AMUD_1 = np.array([0.8926242075417242, 0.8564133234315932, 0.8861344143470382, 0.8895655777207965, 0.8854504171912643, 0.8412944123047621,
          0.8777063733278021, 0.8820057865265066, 0.88979096880361, 0.8911650095245041, 0.8893970215708361, 0.8860786314483806,
          0.8432439984432204, 0.8336446394530547])
AMSD2_1 = np.zeros_like(L0_1)
AMSD1_1 = np.zeros_like(L0_1)
AMSD0_1 = np.zeros_like(L0_1)

L2_5 = np.array([0.34914396097129574, 0.36379415159975637, 0.08323419208719886, 0.09266311152748732, 0.1052541366201737, 0.43776821075426475,
        0.24897503950059643, 0.32113502978668534, 0.39795910262053924, 0.4002002983510159, 0.09684151849507225, 0.10400234067110893,
        0.7438027083607652, 0.8108510821040679])
L1_5 = np.array([206.70308547585122, 266.5638115078195, 102.92117169652055, 122.32414724575992, 138.83319157884958, 309.57734273776623,
        142.24244628804342, 189.01546650249583, 240.02274117037962, 242.1153957801063, 131.77307469423988, 142.82550509460899,
        560.0768072258167, 666.2119999241545])
L0_5 = np.array([74980.64, 126028.22, 65418.576, 73449.328, 75281.353, 116024.839,
        66734.967, 80374.982, 80863.186, 84175.971, 80036.272, 82850.164,
        152271.886, 191762.095])
Linf_5 = np.array([0.005460232271446552, 0.004162398319166094, 0.0025584704788038275, 0.0029130614965538508, 0.0033585062164999224, 0.005635118905079115,
          0.002829040556226484, 0.004919586134142905, 0.006838550525295504, 0.007132547986774604, 0.0032467460287737867, 0.0034734169875310205,
          0.01257949497136899, 0.012776927130400994])
AMUD_5 = np.array([0.9188238451228582, 0.9073859794641642, 0.9092322350590424, 0.9168088969959228, 0.9163296125052316, 0.8960720112448923,
          0.9117749356912775, 0.919255935084821, 0.9212300967399619, 0.9226589699715502, 0.9235606412815969, 0.923138554524134,
          0.9109944882654809, 0.9075682811408905])
AMSD2_5 = np.zeros_like(L0_1)
AMSD1_5 = np.zeros_like(L0_1)
AMSD0_5 = np.zeros_like(L0_1)


def get_min_max(X, batch_size=1000):
    index = 0
    minimum = np.Inf
    maximum = -np.Inf
    while index < len(X):
        new_index = min(len(X), index + batch_size)
        batch = X[index:new_index]
        minimum = min(minimum, np.min(batch))
        maximum = max(maximum, np.max(batch))
        index = new_index
    return minimum, maximum


if __name__ == '__main__':
    os.chdir('../../../')

    with open(data_filename, 'rb') as handle:
        selected_images = pickle.load(handle)
    image_filenames = selected_images['filenames']
    image_labels = selected_images['labels']
    network_names = []

    for i, (network, preprocess, type) in enumerate(networks):

        network_name = network.__name__
        network_names.append(network_name)

        input_shape = getsize(network_name)

        image_generator = FromDiskGenerator(
            image_filenames, target_size=input_shape, batch_size=image_batch_size, preprocess_func=preprocess
        )

        minimum, maximum = get_min_max(image_generator)

        L2_1[i] *= (maximum - minimum)
        L2_5[i] *= (maximum - minimum)
        R = np.prod(input_shape)
        AMSD0_1[i] = L0_1[i] / R
        AMSD1_1[i] = L1_1[i] / R
        AMSD2_1[i] = L2_1[i] / np.sqrt(R)

        AMSD0_5[i] = L0_5[i] / R
        AMSD1_5[i] = L1_5[i] / R
        AMSD2_5[i] = L2_5[i] / np.sqrt(R)

    index = np.argsort(L2_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(L2_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(L2_5)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet $l_2$ Results')
    ax.set_xlabel('$l_2$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 5'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(L1_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(L1_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(L1_5)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet $l_1$ Results')
    ax.set_xlabel('$l_1$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 5'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(L0_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(L0_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(L0_5)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet $l_0$ Results')
    ax.set_xlabel('$l_0$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 5'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(Linf_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(Linf_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(Linf_5)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet $l_{\infty}$ Results')
    ax.set_xlabel('$l_{\infty}$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 5'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(AMSD0_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(AMSD0_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(AMSD0_5)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet U-AMSD$^N_0$ Results')
    ax.set_xlabel('U-AMSD$^N_0$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 5'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(AMSD1_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(AMSD1_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(AMSD1_5)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet U-AMSD$^N_1$ Results')
    ax.set_xlabel('U-AMSD$^N_1$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 5'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(AMSD2_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(AMSD2_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(AMSD2_5)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet U-AMSD$^N_2$ Results')
    ax.set_xlabel('U-AMSD$^N_2$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 5'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(AMUD_1)[::-1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(AMUD_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(AMUD_5)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet U-AMUD$^N$ Results')
    ax.set_xlabel('U-AMUD$^N$')
    ax.set_xlim(.8, 1.)
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 5'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()