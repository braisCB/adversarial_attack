from keras import backend as K, losses
import numpy as np


class AdversarialRankN:

    def __init__(self, model):
        self.model = model
        self.adversarial_func = None

    def build(self):
        lp = K.learning_phase()
        gradient = K.gradients(self.gain_function(
            self.model.targets[0], self.model.outputs[0]), self.model.inputs)
        self.adversarial_func = K.function([self.model.inputs[0], self.model.targets[0], lp], [gradient[0], self.model.outputs[0]])

    def get_adversarial_scores(
            self, X, y, Ns, constraint=None, batch_size=10, alpha=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        if self.adversarial_func is None:
            self.build()

        is_int = isinstance(Ns, int)
        Ns = np.asarray([Ns]) if is_int else Ns
        scores = {n: {'dist': np.zeros(len(X)), 'entropy': np.zeros(len(X))} for n in Ns}

        active_indexes = np.arange(min(len(X), batch_size)).astype(int)
        inactive_indexes = np.arange(min(len(X), batch_size), len(X)).astype(int)
        y_argmax = np.asarray(y).argmax(axis=-1)

        X_adversarial = X[active_indexes].copy()

        v_dX = np.zeros_like(X_adversarial)
        s_dX = np.zeros_like(X_adversarial)

        iters = np.zeros(batch_size)

        batch_not_computed = np.ones((len(Ns), batch_size), dtype=bool)

        cont = 0
        while len(active_indexes):
            iters += 1.
            if cont % 100 == 0:
                print('Cont : ', cont, ', Remaining : ', len(inactive_indexes) + len(active_indexes))

            gradient, output = self.adversarial_func([X_adversarial, y[active_indexes], 0])

            rank_output = (-1. * output).argsort(axis=-1)
            y_output = output[range(len(output)), y_argmax[active_indexes]]

            for i, n in enumerate(Ns):
                # pos_output = (rank_output == n).argmax(axis=-1)
                y_thresh = output[range(len(output)), rank_output[:, n]]
                completed = np.where((y_thresh >= y_output) & batch_not_computed[i])[0]
                if len(completed):
                    pos = active_indexes[completed]
                    scores[n]['dist'][pos] = self.compute_dist(X[pos], X_adversarial[completed])
                    scores[n]['entropy'][pos] = self.compute_entropy(X[pos], X_adversarial[completed])
                    batch_not_computed[i, completed] = False

            incompleted = np.where(batch_not_computed.sum(axis=0) > 0)[0]

            v_dX[incompleted] = beta1 * v_dX[incompleted] + (1. - beta1) * gradient[incompleted]
            s_dX[incompleted] = beta2 * s_dX[incompleted] + (1. - beta2) * np.square(gradient[incompleted])

            v_dX_c = v_dX[incompleted] / (1. - np.power(beta1, iters[incompleted]))
            s_dX_c = s_dX[incompleted] / (1. - np.power(beta2, iters[incompleted]))

            X_adversarial[incompleted] -= alpha * v_dX_c / (np.sqrt(s_dX_c) + epsilon) # / np.max(np.abs(gradient), axis=-1, keepdims=True)
            if constraint is not None:
                X_adversarial[incompleted] = constraint(X_adversarial[incompleted])

            if len(incompleted) != batch_size:
                active_indexes = active_indexes[incompleted]
                batch_not_computed = batch_not_computed[:, incompleted]
                X_adversarial = X_adversarial[incompleted]
                v_dX = v_dX[incompleted]
                s_dX = s_dX[incompleted]
                iters = iters[incompleted]
                nslots = min(len(inactive_indexes), batch_size - len(active_indexes))
                if nslots:
                    active_indexes = np.concatenate((active_indexes, inactive_indexes[:nslots]))
                    batch_not_computed = np.concatenate(
                        (batch_not_computed, np.ones((len(Ns), nslots), dtype=bool)), axis=1
                    )
                    v_dX = np.concatenate(
                        (v_dX, np.zeros((nslots, ) + X_adversarial.shape[1:], dtype=bool)), axis=0
                    )
                    s_dX = np.concatenate(
                        (s_dX, np.zeros((nslots,) + X_adversarial.shape[1:], dtype=bool)), axis=0
                    )
                    iters = np.concatenate((iters, np.zeros(nslots)))
                    X_adversarial = np.concatenate((X_adversarial, X[inactive_indexes[:nslots]].copy()), axis=0)
                    inactive_indexes = inactive_indexes[nslots:]
            cont += 1
        for i in scores:
            scores[i]['dist'] = scores[i]['dist'].tolist()
            scores[i]['entropy'] = scores[i]['entropy'].tolist()
        return scores[Ns[0]] if is_int else scores

    @classmethod
    def gain_function(cls, y_true, y_pred):
        y_pred = cls.clip(K.epsilon(), 1. - K.epsilon())(y_pred)
        # return -1. * K.sum(y_true * K.log(1. - y_pred), axis=-1) - K.sum(K.log(1. - K.relu(K.max(y_pred, axis=-1, keepdims=True) - (1. - y_true) * y_pred)), axis=-1)
        return -1. * K.sum(y_true * K.log(1. - y_pred) + (1. - y_true) * K.log(y_pred), axis=-1)

    @staticmethod
    def compute_dist(X, X_adversarial):
        return np.linalg.norm((X - X_adversarial).reshape((-1, np.prod(X.shape[1:]))), axis=1)

    @staticmethod
    def compute_entropy(X, X_adversarial):
        diff = np.square(X - X_adversarial).reshape((-1, np.prod(X.shape[1:])))
        nfeats = diff.shape[-1]
        diff /= np.maximum(1e-6, diff.sum(axis=1, keepdims=True))
        diff = diff * np.log(diff)
        diff[np.isinf(diff) | np.isnan(diff)] = 0.
        return -1. * diff.sum(axis=1) / np.log(nfeats)

    @staticmethod
    def clip(min_value, max_value):
        @K.tf.custom_gradient
        def clip_by_value(x):
            # x_clip = K.clip(x - bias, -2., 2.)
            s = K.clip(x, min_value, max_value)

            def grad(dy):
                return dy

            return s, grad

        return clip_by_value
