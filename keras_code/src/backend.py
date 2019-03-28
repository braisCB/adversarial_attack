from keras import backend as K


def custom_softmax(eps=.95):

    def softmax_func(x, axis=-1):
        exp_x = K.exp(x)
        nelems = K.int_shape(x)[axis]
        sum_exp_x = K.sum(exp_x, axis=axis, keepdims=True)
        return eps * exp_x / sum_exp_x + (1. - eps) * 1. / nelems

    return softmax_func


K.classic_softmax = K.softmax
K.softmax = custom_softmax()
