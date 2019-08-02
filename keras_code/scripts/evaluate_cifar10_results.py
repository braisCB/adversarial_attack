import pickle
import numpy as np
import matplotlib.pyplot as plt

filenames = [
    ('pixeldefend', 'info/pixeldefend.pickle'),
    ('lid', 'info/lid.pickle'),
    # ('thermometer', 'info/thermometer_new.pickle'),
    ('cascade', 'info/cascade.pickle'),
    ('sap', 'info/sap.pickle'),
    ('madry', 'info/madry.pickle')
]

if __name__ == '__main__':
    l_inf_data = []
    defenses = []
    for name, filename in filenames:
        with open(filename, 'rb') as outfile:
            info_data = pickle.load(outfile)

        key = list(info_data.keys())[0] if name != 'madry' else 'adv_trained'
        nelements = np.sqrt(24 * 24) if name == 'cascade' else (32 * 32 * 24 if name == 'thermometer' else np.sqrt(32 * 32)) # for cascade 24*24 else 32*32
        defenses.append(name)
        # l_inf_data.append(np.array(info_data[key]['DoS'][1]['dist']) / nelements)
        l_inf_data.append(np.array(info_data[key]['DoS'][1]['entropy']))

        print(info_data.keys(), nelements)
        print('DEFENSE : ', name)
        print('DoS N = 3')
        print('AMSD : ', np.mean(np.array(info_data[key]['DoS'][3]['dist'])) / nelements)
        print('AMUD : ', np.mean(info_data[key]['DoS'][3]['entropy']))
        print('DoS N = 1')
        print('AMSD : ', np.mean(np.array(info_data[key]['DoS'][1]['dist'])) / nelements)
        print('AMUD : ', np.mean(info_data[key]['DoS'][1]['entropy']))
        print('Phishing N = 1')
        print('AMSD(0.5) : ', np.mean(np.array(info_data[key]['phishing'][1][0.5]['dist'])) / nelements)
        print('AMUD(0.5) : ', np.mean(info_data[key]['phishing'][1][0.5]['entropy']))
        print('AMSD(0.75) : ', np.mean(np.array(info_data[key]['phishing'][1][0.75]['dist'])) / nelements)
        print('AMUD(0.75) : ', np.mean(info_data[key]['phishing'][1][0.75]['entropy']))
        print('AMSD(0.95) : ', np.mean(np.array(info_data[key]['phishing'][1][0.95]['dist'])) / nelements)
        print('AMUD(0.95) : ', np.mean(info_data[key]['phishing'][1][0.95]['entropy']))

        # l_inf = np.array(info_data[key]['DoS'][1]['dist']) / nelements
        try:
            l_inf = np.array(info_data[key]['DoS'][1]['dist_inf'])
            print('MEAN_ATTACK : ', (l_inf < 8./255.).mean())
            print('M = ', l_inf.mean())
        except:
            pass
        if 'accuracy' in info_data[key]:
            print('ACCURACY : ', info_data[key]['accuracy'])

    l_inf_data = np.array(l_inf_data)
    values = np.arange(0.85, 1, 1e-4) # np.sort(l_inf_data.flatten())#

    print('MAX: ', values.max())

    results = np.zeros((len(l_inf_data), len(values)))
    for i, v in enumerate(values):
        results[:, i] = (l_inf_data > v).mean(axis=-1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for result in results:
        ax.plot(values, result)
    ax.set_title('CIFAR-10 DoS Error Rate')


    ax.set_xlabel('AMUD$_{DoS}(N = 1)$')
    ax.set_ylabel('Error Rate')
    ax.legend(defenses)

    plt.tight_layout()

    plt.show()




