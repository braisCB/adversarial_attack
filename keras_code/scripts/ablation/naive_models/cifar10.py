import os
from keras.datasets import cifar10
from keras_code.scripts.ablation.models import naive_densenet, naive_inception, naive_resnet
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras_code.src.AdversarialRankN import AdversarialRankN
from keras_code.src.AdversarialPhishingN import AdversarialPhishingN
from keras.utils.vis_utils import plot_model
import json


shape = (32, 32, 3)
nclasses = 10
filename = 'naive_models.json'
info_folder = './keras_code/scripts/ablation/info/'
batch_size = 128
epochs = 100
Ns = [3, 1]
alpha = 1e-4
threshs = [.5, .75, .9, .95]

generator = ImageDataGenerator(
    width_shift_range=5. / 32,
    height_shift_range=5. / 32,
    fill_mode='reflect',
    horizontal_flip=True
)


def scheduler(adam=True):
    def sch(epoch):
        if epoch < 30:
            lr = .1
        elif epoch < 60:
            lr = .02
        elif epoch < 80:
            lr = .004
        else:
            lr = .0008
        if adam:
            lr /= 10.
        print('lr: ', lr)
        return lr
    return sch


def boundary_constraint(minval, maxval):

    def func(x):
        x[x < minval] = minval
        x[x > maxval] = maxval
        return x

    return func


if __name__ == '__main__':
    os.chdir('../../../../')

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.] = 1.

    x_train = 2. * x_train / 255. - 1.
    x_test = 2. * x_test / 255. - 1.

    y_train = to_categorical(y_train, nclasses)
    y_test = to_categorical(y_test, nclasses)


    for naive_model in [naive_resnet.NaiveResNet, naive_inception.NaiveInception, naive_densenet.NaiveDenseNet]:

        for global_average_pooling in [False]:
            name = naive_model.__name__ + '_gap_' + str(global_average_pooling)
            print('model : ', name)

            model = naive_model(shape=shape, nclasses=nclasses, gap=global_average_pooling).model
            model.summary()

            plot_model(model, to_file=name + '.png', show_shapes=True, show_layer_names=True)
            continue

            model.fit_generator(
                generator.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs,
                callbacks=[LearningRateScheduler(scheduler())],
                validation_data=(x_test, y_test),
                validation_steps=x_test.shape[0] // batch_size,
                verbose=2
            )

            adversarial_rank = AdversarialRankN(model=model)
            rank_scores = adversarial_rank.get_adversarial_scores(
                x_test, y_test, Ns=Ns, batch_size=batch_size, alpha=alpha, constraint=boundary_constraint(-1., 1.)
            )

            adversarial_phishing = AdversarialPhishingN(model=model)
            phishing_scores = adversarial_phishing.get_adversarial_scores(
                x_test, y_test, threshs=threshs, Ns=Ns, batch_size=batch_size, alpha=alpha,
                constraint=boundary_constraint(-1., 1.)
            )

            if not os.path.isdir(info_folder):
                os.makedirs(info_folder)

            try:
                with open(info_folder + filename) as outfile:
                    info_data = json.load(outfile)
            except:
                info_data = {}

            info_data[name] = {
                'DoS': rank_scores, 'phishing': phishing_scores
            }

            with open(info_folder + filename, 'w') as outfile:
                json.dump(info_data, outfile)



