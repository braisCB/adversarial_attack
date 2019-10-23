from keras import backend as K


def custom_softmax(factor=15.):

    def softmax_func(x, axis=-1):
        max_x = K.max(x, axis=axis, keepdims=True)
        x = x - max_x
        min_x = -1. * (K.min(x, axis=axis, keepdims=True) - 1)
        x = factor * x / K.maximum(factor, min_x)
        exp_x = K.exp(x)
        # exp_x = K.tf.Print(exp_x, [K.max(min_x)])
        sum_exp_x = K.sum(exp_x, axis=axis, keepdims=True)
        return exp_x / K.maximum(1e-6, sum_exp_x) # + (1. - eps) * 1. / nelems

    return softmax_func


K.classic_softmax = K.softmax
K.softmax = custom_softmax()
