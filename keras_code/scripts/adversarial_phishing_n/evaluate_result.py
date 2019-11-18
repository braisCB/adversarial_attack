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


L2_1 = np.array([0.2844903280572445, 0.25670300164502685, 0.15355208184640393, 0.12527911981441833, 0.15073243969249744,
                 0.30146684356488784, 0.23640907331285732, 0.27496867659460056, 0.2865321944906368, 0.28381011121774247,
                 0.10959208910449454, 0.1081702435631411, 0.4328137599202618, 0.45351121499037833])
L1_1 = np.array([109.27805907453197, 112.25895892774676, 182.7706124532891, 142.93518096892663, 178.6495675703826,
                 121.77123682578775, 89.3503793991748, 110.86344113898286, 111.20080431061194, 111.69588341242793,
                 117.26114337019973, 113.29571932604749, 186.29871327644707, 204.90903815395194])
L0_1 = np.array([45056.303, 56173.234, 80974.454, 71737.02, 71522.107,
                 50631.203, 46023.91, 52254.404, 45238.032, 46058.672,
                 60834.206, 58864.96, 56124.726, 66739.367])
Linf_1 = np.array([0.004153720348248851, 0.002764143083717301, 0.005897287285795258, 0.004366777352630934, 0.004954586455814041,
                   0.00314089482398004, 0.0028580705682860826, 0.0044568819015322935, 0.004043568287588766, 0.003938607062622817,
                   0.0039031518870915826, 0.003777619487928612, 0.006759560824057369, 0.007093069556795752])
AMUD_1 = np.array([0.858660739634423, 0.8417625956569368, 0.9026012507621821, 0.8953410645171294, 0.8893812550975515,
                   0.8296449805026279, 0.8629353591570957, 0.8676369217078711, 0.859008677646935, 0.8628965462808215,
                   0.8795435844328225, 0.8768985781300016, 0.8290434943001177, 0.8191274853044324])
AMSD2_1 = np.zeros_like(L0_1)
AMSD1_1 = np.zeros_like(L0_1)
AMSD0_1 = np.zeros_like(L0_1)

L2_100 = np.array([0.6778046993296922, 0.5818705002270289, 0.2592680127821768, 0.214682554267875, 0.2614576078067344,
                   0.7058478233972909, 0.5810668844979527, 0.6730906272706045, 0.7005689155072439, 0.6854207665878786,
                   0.19749598368403853, 0.19954308291783818, 0.8962855576656821, 0.7736025441658978])
L1_100 = np.array([299.1682600495126, 287.13739400484394, 342.6584589825735, 266.72303418097886, 338.321317332268,
                   331.92333793452616, 249.0440748693059, 300.49116499465197, 309.64031608846415, 302.4799479495553,
                   236.86056422079767, 237.61531078804947, 428.22342162622226, 380.53260982530117])
L0_100 = np.array([67751.922, 85155.492, 97370.895, 85698.23, 92535.708,
                   77611.429, 64023.671, 70700.099, 68232.437, 68832.461,
                   81242.834, 80951.871, 83903.938, 88976.752])
Linf_100 = np.array([0.011954064240531418, 0.008229673401802008, 0.009066491997174837, 0.007341870830172205, 0.008715659352780179,
                     0.009686991378182089, 0.009810930395395264, 0.013192836383646529, 0.012357030617137643, 0.012329632426349508,
                     0.007245041642186432, 0.007128489320720375, 0.01643388185627612, 0.01344769378002064])
AMUD_100 = np.array([0.8837047475000945, 0.8574743690349248, 0.9149185684326526, 0.9069638598899016, 0.9127703614147701,
                     0.8526678178824295, 0.8803150350104224, 0.8822367258304341, 0.8838283772400477, 0.8837638165819629,
                     0.9003951436943175, 0.9008015085042342, 0.8509853860454156, 0.84199100120674])
AMSD2_100 = np.zeros_like(L0_1)
AMSD1_100 = np.zeros_like(L0_1)
AMSD0_100 = np.zeros_like(L0_1)


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
        L2_100[i] *= (maximum - minimum)
        R = np.prod(input_shape)
        AMSD0_1[i] = L0_1[i] / R
        AMSD1_1[i] = L1_1[i] / R
        AMSD2_1[i] = L2_1[i] / np.sqrt(R)

        AMSD0_100[i] = L0_100[i] / R
        AMSD1_100[i] = L1_100[i] / R
        AMSD2_100[i] = L2_100[i] / np.sqrt(R)

    index = np.argsort(L2_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(L2_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(L2_100)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet $l_2$ Results')
    ax.set_xlabel('$l_2$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 100'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(L1_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(L1_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(L1_100)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet $l_1$ Results')
    ax.set_xlabel('$l_1$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 100'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(L0_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(L0_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(L0_100)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet $l_0$ Results')
    ax.set_xlabel('$l_0$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 100'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(Linf_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(Linf_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(Linf_100)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet $l_{\infty}$ Results')
    ax.set_xlabel('$l_{\infty}$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 100'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(AMSD0_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(AMSD0_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(AMSD0_100)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet T-AMSD$^N_0$ Results')
    ax.set_xlabel('T-AMSD$^N_0$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 100'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(AMSD1_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(AMSD1_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(AMSD1_100)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet T-AMSD$^N_1$ Results')
    ax.set_xlabel('T-AMSD$^N_1$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 100'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(AMSD2_1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(AMSD2_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(AMSD2_100)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet T-AMSD$^N_2$ Results')
    ax.set_xlabel('T-AMSD$^N_2$')
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 100'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    index = np.argsort(AMUD_1)[::-1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(AMUD_1)[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(AMUD_100)[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_title('ImageNet T-AMUD$^N$ Results')
    ax.set_xlabel('T-AMUD$^N$')
    ax.set_xlim(.8, 1.)
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), ('N = 1', 'N = 100'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()