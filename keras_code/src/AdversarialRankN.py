from keras import backend as K, layers
import numpy as np
from keras_code.src.AdversarialModule import AdversarialModule


class AdversarialRankN(AdversarialModule):

    def __init__(self, model):
        super(AdversarialRankN, self).__init__(model)
        self.adversarial_func = None

    def build(self, n):
        lp = K.learning_phase()
        targets = layers.Input(shape=self.model.output_shape[1:])
        gradient = K.gradients(self.gain_function(
            self.model.targets[0], self.model.outputs[0], targets, n+1), self.model.inputs)
        self.adversarial_func = K.function(
            [self.model.inputs[0], self.model.targets[0], targets, lp], [gradient[0], self.model.outputs[0]]
        )

    def get_adversarial_scores(
            self, X, y, Ns, constraint=None, batch_size=10, alpha=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8,
            l2=1., l1=0., extra_epochs=200
    ):
        is_int = isinstance(Ns, int)
        Ns = np.asarray([Ns]) if is_int else Ns
        scores = {}
        for n in Ns:
            self.build(n)
            scores[n] = self.get_adversarial_scores_for_targets(
                X, y, n, constraint=constraint, batch_size=batch_size, alpha=alpha, beta1=beta1,
                beta2=beta2, epsilon=epsilon, l1=l1, l2=l2, extra_epochs=extra_epochs
            )
        return scores

    def get_adversarial_scores_for_targets(
            self, X, y, n, constraint=None, batch_size=10, alpha=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8,
            l2=1e-3, l1=0., extra_epochs=200
    ):

        diff_image = np.zeros(self.model.input_shape[1:])
        square_diff_image = np.zeros(self.model.input_shape[1:])
        count_finished = 0
        scores = {
            'amsd': np.Inf * np.ones(len(X)), 'amud': np.zeros(len(X)),
            'mean': np.zeros(len(X)), 'variance': np.zeros(len(X)), 'zero_variance': np.zeros(len(X))
        }

        active_indexes = np.arange(min(len(X), batch_size)).astype(int)
        inactive_indexes = np.arange(min(len(X), batch_size), len(X)).astype(int)
        y_argmax = np.asarray(y).argmax(axis=-1)

        X_active = X[active_indexes]
        X_adversarial = X_active.copy()
        active_targets = np.zeros_like(y[active_indexes])
        X_min = X_active.min()
        X_max = X_active.max()

        v_dX = np.zeros_like(X_adversarial)
        s_dX = np.zeros_like(X_adversarial)

        iters = np.zeros(batch_size)
        extra_iters = np.zeros(batch_size)

        ndims = [1] * X_active.ndim
        ndims[0] = -1

        cont = 0
        while len(active_indexes):
            first_iter = np.where(iters == 0.)[0]
            if len(first_iter) > 0:
                active_targets[first_iter] = self.get_target(
                    X_active[first_iter], y_argmax[active_indexes[first_iter]], n
                )

            iters += 1.

            gradient, output = self.adversarial_func([X_adversarial, y[active_indexes], active_targets, 0])

            y_output = output[range(len(output)), y_argmax[active_indexes]]
            y_thresh = np.min(output / np.maximum(1e-8, active_targets), axis=-1)
            completed = np.where(y_thresh >= y_output)[0]
            extras = np.where((y_thresh >= y_output) | (extra_iters > 0))[0]
            if len(extras) > 0:
                if l2 > 0.:
                    gradient[extras] += l2 * (X_adversarial[extras] - X_active[extras])
                if l1 > 0.:
                    gradient[extras] += l1 * np.sign(X_adversarial[extras] - X_active[extras])
                extra_iters[extras] += 1
            if len(completed) > 0:
                pos = active_indexes[completed]
                diff = X_adversarial[completed] - X_active[completed]
                amsd = self.compute_amsd(diff)
                new_winner = np.where(amsd < scores['amsd'][pos])[0]
                if len(new_winner) > 0:
                    diff = diff[new_winner]
                    pos = pos[new_winner]
                    scores['amsd'][pos] = amsd[new_winner]
                    scores['amud'][pos] = self.compute_amud(diff)
                    scores['mean'][pos] = self.compute_mean(diff)
                    scores['variance'][pos] = self.compute_variance(diff)
                    scores['zero_variance'][pos] = self.compute_zero_variance(diff)
                diff_image += np.sum(diff, axis=0)
                square_diff_image += np.sum(np.square(diff), axis=0)
                count_finished += len(completed)

            incompleted = np.where((extra_iters < extra_epochs) & ((extra_iters != 1) | (iters != 1)))[0]

            v_dX[incompleted] = beta1 * v_dX[incompleted] + (1. - beta1) * gradient[incompleted]
            s_dX[incompleted] = beta2 * s_dX[incompleted] + (1. - beta2) * np.square(gradient[incompleted])

            v_dX_c = v_dX[incompleted] # / (1. - np.power(beta1, iters[incompleted]).reshape(ndims))
            s_dX_c = s_dX[incompleted] # / (1. - np.power(beta2, iters[incompleted]).reshape(ndims))

            X_adversarial[incompleted] -= self.get_alpha(alpha, iters[incompleted]).reshape(ndims) * v_dX_c / (np.sqrt(s_dX_c) + epsilon) # / np.max(np.abs(gradient), axis=-1, keepdims=True)
            if constraint is not None:
                X_adversarial[incompleted] = constraint(X_adversarial[incompleted])

            if len(incompleted) != batch_size:
                active_indexes = active_indexes[incompleted]
                active_targets = active_targets[incompleted]
                X_adversarial = X_adversarial[incompleted]
                X_active = X_active[incompleted]
                v_dX = v_dX[incompleted]
                s_dX = s_dX[incompleted]
                iters = iters[incompleted]
                extra_iters = extra_iters[incompleted]
                nslots = min(len(inactive_indexes), batch_size - len(active_indexes))
                if nslots:
                    active_indexes = np.concatenate((active_indexes, inactive_indexes[:nslots]))
                    active_targets = np.concatenate(
                        (active_targets, np.zeros((nslots,) + active_targets.shape[1:])), axis=0
                    )
                    v_dX = np.concatenate(
                        (v_dX, np.zeros((nslots, ) + X_adversarial.shape[1:], dtype=bool)), axis=0
                    )
                    s_dX = np.concatenate(
                        (s_dX, np.zeros((nslots,) + X_adversarial.shape[1:], dtype=bool)), axis=0
                    )
                    iters = np.concatenate((iters, np.zeros(nslots)))
                    extra_iters = np.concatenate((extra_iters, np.zeros(nslots)))
                    X_active = np.concatenate((X_active, X[inactive_indexes[:nslots]]), axis=0)
                    X_min = min(X_min, X_active.min())
                    X_max = max(X_max, X_active.max())
                    X_adversarial = np.concatenate((X_adversarial, X_active[-nslots:].copy()), axis=0)
                    inactive_indexes = inactive_indexes[nslots:]
            if cont % 100 == 0:
                print('Cont : ', cont, ', Remaining : ', len(inactive_indexes) + len(active_indexes), ', Max iter : ', iters.max(), ', Max extra : ', extra_iters.max(), ' Max_output : ', y_output.max(), ' Min_thresh : ', y_thresh.min())
            cont += 1
        scores['amsd'] = (scores['amsd'] / (X_max - X_min)).tolist()
        scores['amud'] = scores['amud'].tolist()
        scores['mean'] = scores['mean'].tolist()
        scores['variance'] = scores['variance'].tolist()
        scores['zero_variance'] = scores['zero_variance'].tolist()
        scores['min'] = X_min
        scores['max'] = X_max
        mean = (diff_image / count_finished)
        variance = square_diff_image / count_finished - np.square(mean)
        scores['image_mean'] = mean.tolist()
        scores['image_variance'] = variance.tolist()
        print('AMSD: ', np.mean(scores['amsd']))
        print('AMUD: ', np.mean(scores['amud']))
        return scores

    def gain_function(self, y_true, y_pred, y_target, factor, epsilon=1e-3):
        y_pred_clipped = K.clip(y_pred * (factor - epsilon), 0., 1.)
        return -1 * K.sum(y_target * K.log(y_pred_clipped), axis=-1)

    def get_target(self, X, y_argmax, n):
        y_pred = self.model.predict(X)
        y_pred_argsort = np.argsort(-1. * y_pred, axis=-1)
        y_targets = y_pred_argsort[y_pred_argsort != y_argmax.reshape((-1, 1))].reshape((y_argmax.shape[0], -1))
        targets = np.zeros_like(y_pred)
        for i, y_target in enumerate(y_targets):
            targets[i, y_target[:n]] = 1.
        return targets

