import numpy as np


class AdversarialModule:

    def __init__(self, model):
        self.model = model

    def compute_dist(self, diff):
        return np.linalg.norm(diff.reshape((-1, np.prod(diff.shape[1:]))), axis=1)

    def compute_mean(self, diff):
        return np.mean(diff.reshape((-1, np.prod(diff.shape[1:]))), axis=1)

    def compute_variance(self, diff):
        return np.var(diff.reshape((-1, np.prod(diff.shape[1:]))), axis=1)

    def compute_zero_variance(self, diff):
        return np.mean(np.square(diff).reshape((-1, np.prod(diff.shape[1:]))), axis=1)

    @staticmethod
    def compute_entropy(diff):
        diff = np.square(diff).reshape((-1, np.prod(diff.shape[1:])))
        diff = np.clip(diff, 1e-8, 1. - 1e-8)
        nfeats = diff.shape[-1]
        diff /= diff.sum(axis=1, keepdims=True)
        diff = diff * np.log(diff)
        return -1. * diff.sum(axis=1) / np.log(nfeats)

    @staticmethod
    def get_alpha(alpha, y_output):
        new_alpha = alpha * np.ones(y_output.shape[0])
        # new_alpha[y_output > .95] *= 2.
        # new_alpha[y_output > .99] *= 2.
        return new_alpha
