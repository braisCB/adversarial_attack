import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import friedmanchisquare


filename = './info/mnist_naive_models_fgsm.pickle'
tipo = 'DoS'


if __name__ == '__main__':

    try:
        with open(filename) as outfile:
            info_data = json.load(outfile)
    except:
        with open(filename, 'rb') as outfile:
            info_data = pickle.load(outfile)

    network_names = []
    amsd_results = {}
    ame_results = {}
    for network in info_data:
        network_dict = info_data[network][tipo]
        network_names.append(network)
        print('NETWORK : ', network)
        print('ACC: ', info_data[network]['evaluation'][-1])
        for n in network_dict:
            print('N : ', n)
            if n not in amsd_results:
                amsd_results[n] = []
            # network_dict[n]['dist'] = np.asarray(network_dict[n]['dist'])
            # network_dict[n]['dist'] = network_dict[n]['dist'][network_dict[n]['dist'] > 0]
            # network_dict[n]['entropy'] = np.asarray(network_dict[n]['entropy'])
            # network_dict[n]['entropy'] = network_dict[n]['entropy'][network_dict[n]['entropy'] > 0]
            network_dict[n]['dist'] = np.array(network_dict[n]['dist']) / (32*32)
            network_dict[n]['dist'] = network_dict[n]['dist'][network_dict[n]['dist'] > 0]
            network_dict[n]['entropy'] = np.array(network_dict[n]['entropy'])
            amsd_results[n].append(np.mean(network_dict[n]['dist']))
            print('AMSD :', amsd_results[n][-1], '+-', np.std(network_dict[n]['dist']))
            if n not in ame_results:
                ame_results[n] = []
            ame_results[n].append(np.mean(network_dict[n]['entropy']))
            print('AMUD :', ame_results[n][-1], '+-', np.std(network_dict[n]['entropy']))
            print('')

    keys = list(sorted(amsd_results.keys()))
    index = np.argsort(amsd_results[keys[-1]])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(amsd_results[keys[0]])[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(amsd_results[keys[1]])[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_xlabel('AMSD')
    ax.set_yticks(ind + 0.5*width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), keys)

    plt.tight_layout()

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
                    ha='center', va='bottom')

    # autolabel(rects2)
    # autolabel(rects3)
    keys = list(sorted(ame_results.keys()))
    index = np.argsort(ame_results[keys[-1]])[::-1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.4

    zvals = np.array(ame_results[keys[0]])[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(ame_results[keys[1]])[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')

    ax.set_xlabel('AME')
    ax.set_xlim(.95, 1.)
    ax.set_yticks(ind + 0.5 * width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0]), keys)

    plt.tight_layout()

    plt.show()