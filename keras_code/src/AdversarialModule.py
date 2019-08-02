import numpy as np


class AdversarialModule:

    def __init__(self, model):
        self.model = model

    def compute_l2(self, diff):
        return np.linalg.norm(diff.reshape((-1, np.prod(diff.shape[1:]))), ord=2, axis=1)
        # return np.mean(np.square(diff).reshape((-1, np.prod(diff.shape[1:]))), axis=1)

    def compute_l1(self, diff):
        return np.linalg.norm(diff.reshape((-1, np.prod(diff.shape[1:]))), ord=1, axis=1)
        # return np.mean(np.abs(diff).reshape((-1, np.prod(diff.shape[1:]))), axis=1)

    def compute_l_inf(self, diff):
        return np.linalg.norm(diff.reshape((-1, np.prod(diff.shape[1:]))), ord=np.inf, axis=1)
        # return np.max(np.abs(diff).reshape((-1, np.prod(diff.shape[1:]))), axis=1)

    def compute_l0(self, diff):
        return np.linalg.norm(diff.reshape((-1, np.prod(diff.shape[1:]))), ord=0, axis=1)
        # return np.mean(np.square(diff).reshape((-1, np.prod(diff.shape[1:]))), axis=1)

    @staticmethod
    def compute_amud(diff):
        diff = np.square(diff).reshape((-1, np.prod(diff.shape[1:])))
        diff = np.clip(diff, 1e-8, 1. - 1e-8)
        nfeats = diff.shape[-1]
        diff /= diff.sum(axis=1, keepdims=True)
        diff = diff * np.log(diff)
        return -1. * diff.sum(axis=1) / np.log(nfeats)

    @staticmethod
    def get_alpha(alpha, iters):
        new_alpha = alpha * np.ones(iters.shape[0])
        new_alpha[iters > 3500] *= 5.
        new_alpha[iters > 2000] *= 5.
        # new_alpha[iters > 500] *= 5.
        # new_alpha[iters > 200] *= 5.
        # new_alpha[y_output > .99] *= 2.
        return new_alpha
