import json
import numpy as np
import matplotlib.pyplot as plt


filename = './info/phishing_imagenet_rank.json'


def getsize(network_name):
    if network_name == 'NASNetLarge':
        return 331*331*3
    if network_name in ['InceptionResNetV2', 'InceptionV3', 'Xception']:
        return 299*299*3
    else:
        return 224*224*3


if __name__ == '__main__':

    with open(filename) as outfile:
        info_data = json.load(outfile)

    network_names = []
    amsd_results = {}
    ame_results = {}
    for network in info_data:
        network_dict = info_data[network]['1']
        network_names.append(network)
        for n in network_dict:
            print('NETWORK : ', network)
            print('N : ', n)
            if n not in amsd_results:
                amsd_results[n] = []
            # network_dict[n]['dist'] = np.asarray(network_dict[n]['dist'])
            # network_dict[n]['dist'] = network_dict[n]['dist'][network_dict[n]['dist'] > 0]
            # network_dict[n]['entropy'] = np.asarray(network_dict[n]['entropy'])
            # network_dict[n]['entropy'] = network_dict[n]['entropy'][network_dict[n]['entropy'] > 0]
            amsd_results[n].append(np.mean(network_dict[n]['dist']) / np.sqrt(getsize(network)))
            print('MSD_{DoS}^k :', amsd_results[n][-1])
            if n not in ame_results:
                ame_results[n] = []
            ame_results[n].append(np.mean(network_dict[n]['entropy']))
            print('ADE_{DoS}^k :', ame_results[n][-1])
            print('')

    index = np.argsort(amsd_results['0.95'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.3

    zvals = np.array(amsd_results['0.5'])[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(amsd_results['0.75'])[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')
    zvals = np.array(amsd_results['0.95'])[index]
    rects4 = ax.barh(ind + 2*width, zvals, width, color='m')

    ax.set_title('ImageNet Phishing Results')
    ax.set_xlabel('$AMSD_{Ph}^2$')
    ax.set_yticks(ind + width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0], rects4[0]), ('$p_{th} = 0.5$', '$p_{th} = 0.75$', '$p_{th} = 0.95$'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
                    ha='center', va='bottom')

    # autolabel(rects2)
    # autolabel(rects3)

    index = np.argsort(ame_results['0.95'])[::-1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.3

    zvals = np.array(ame_results['0.5'])[index]
    rects2 = ax.barh(ind, zvals, width, color='g')
    kvals = np.array(ame_results['0.75'])[index]
    rects3 = ax.barh(ind + width, kvals, width, color='b')
    zvals = np.array(ame_results['0.95'])[index]
    rects4 = ax.barh(ind + 2*width, zvals, width, color='m')

    ax.set_title('ImageNet Phishing Results')
    ax.set_xlabel('$AMUD_{Ph}$')
    ax.set_xlim(.95, 1.)
    ax.set_yticks(ind + width)
    ax.set_yticklabels(np.array(network_names)[index])
    ax.legend((rects2[0], rects3[0], rects4[0]), ('$p_{th} = 0.5$', '$p_{th} = 0.75$', '$p_{th} = 0.95$'))

    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.show()