import json
import numpy as np
import matplotlib.pyplot as plt


filename = './info/cifar_10_naive_models.json'
tipo = 'evaluation'


if __name__ == '__main__':

    with open(filename) as outfile:
        info_data = json.load(outfile)

    network_names = []
    acc_results = []
    for network in info_data:
        network_dict = info_data[network][tipo]
        network_names.append(network)
        acc_results.append(network_dict[-1])
        print('ACC :', acc_results[-1])


    index = np.argsort(acc_results)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(network_names))
    width = 0.8

    zvals = np.array(acc_results)[index]
    rects2 = ax.barh(ind+width/2, zvals, width, color='g')

    ax.set_xlabel('ACC')
    ax.set_yticks(ind + 0.5*width)
    ax.set_yticklabels(np.array(network_names)[index])

    plt.tight_layout()

    plt.show()


# def important_points(trajectory, thresh=0.75, momentum=.9):
#
#     trajectory = np.asarray(trajectory)
#
#     if len(trajectory) < 3:
#         return trajectory
#
#     output = [trajectory[0]]
#     direction = trajectory[1] - trajectory[0]
#     direction /= np.linalg.norm(direction)
#     last_p = trajectory[1]
#
#     for i, p in trajectory[2:]:
#         new_direction = p - output[1]
#         new_direction /= np.linalg.norm(new_direction)
#         cos = (new_direction * direction).sum()
#         if cos < thresh:
#             output.append(last_p)
#             direction = p - last_p
#         else:
#             direction = momentum * direction + (1. - momentum) * new_direction
#         last_p = p
#         direction /= np.linalg.norm(direction)
#
#     return np.array(output)

