import numpy as np


class AdversarialModule:

    def compute_dist(self, X, X_adversarial):
        return np.linalg.norm((X - X_adversarial).reshape((-1, np.prod(X.shape[1:]))), axis=1)

    @staticmethod
    def compute_entropy(X, X_adversarial):
        diff = np.square(X - X_adversarial).reshape((-1, np.prod(X.shape[1:])))
        diff = np.clip(diff, 1e-8, 1. - 1e-8)
        nfeats = diff.shape[-1]
        diff /= diff.sum(axis=1, keepdims=True)
        diff = diff * np.log(diff)
        return (np.log(nfeats) - diff.sum(axis=1)) / np.log(nfeats)

    @staticmethod
    def get_alpha(alpha, y_output):
        new_alpha = alpha * np.ones_like(y_output)
        # new_alpha[y_output > .95] *= 2.
        # new_alpha[y_output > .99] *= 2.
        return new_alpha
