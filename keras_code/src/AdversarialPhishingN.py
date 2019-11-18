from keras import backend as K, layers
import numpy as np
from keras_code.src.AdversarialModule import AdversarialModule


class AdversarialPhishingN(AdversarialModule):

    def __init__(self, model):
        super(AdversarialPhishingN, self).__init__(model)
        self.adversarial_func = None

    def build(self, thresh):
        lp = K.learning_phase()
        targets = layers.Input(shape=self.model.output_shape[1:])
        gradient = K.gradients(self.gain_function(
            self.model.targets[0], self.model.outputs[0], targets, thresh), self.model.inputs)
        self.adversarial_func = K.function(
            [self.model.inputs[0], self.model.targets[0], targets, lp], [gradient[0], self.model.outputs[0]]
        )

    def get_adversarial_scores(
            self, X, y, n, threshs, constraint=None, batch_size=10, alpha=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8,
            l2=1e-4, l1=1e-4, extra_epochs=250, save_data=None, inverse_transform=None
    ):
        scores = {}
        for thresh in threshs:
            self.build(thresh)
            key = str(n) + '_' + str(thresh)
            scores[key] = self.get_adversarial_scores_for_targets(
                X, y, n, thresh, constraint=constraint, batch_size=batch_size, alpha=alpha, beta1=beta1, beta2=beta2,
                epsilon=epsilon, l2=l2, l1=l1, extra_epochs=extra_epochs, save_data=save_data,
                inverse_transform=inverse_transform
            )
        return scores

    def get_adversarial_scores_for_targets(
            self, X, y, n, thresh, constraint=None, batch_size=10, alpha=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8,
            l2=0., l1=0., extra_epochs=40, save_data=None, inverse_transform=None
    ):
        if not isinstance(n, int):
            n = np.asarray(n)

        diff_image = np.zeros(self.model.input_shape[1:])
        square_diff_image = np.zeros(self.model.input_shape[1:])
        count_finished = 0
        scores = {
            'L2': np.Inf * np.ones(len(X)), 'amud': np.Inf * np.ones(len(X)),
            'L1': np.Inf * np.ones(len(X)), 'L_inf': np.Inf * np.ones(len(X)), 'L0': np.Inf * np.ones(len(X)),
            'AMSD_2': np.Inf * np.ones(len(X)), 'AMSD_1': np.Inf * np.ones(len(X)),
            'AMSD_0': np.Inf * np.ones(len(X)),
        }

        if save_data is not None:
            if not (isinstance(save_data, list) or isinstance(save_data, np.ndarray)):
                scores['adversarial_data'] = np.zeros((len(X),) + self.model.input_shape[1:])
            else:
                save_data = np.asarray(save_data)

        active_indexes = np.arange(min(len(X), batch_size)).astype(int)
        inactive_indexes = np.arange(min(len(X), batch_size), len(X)).astype(int)
        y_argmax = np.asarray(y).argmax(axis=-1)

        X_active = X[active_indexes]
        X_adversarial = X_active.copy()
        X_best = X_active.copy()
        active_targets = np.zeros_like(y[active_indexes])
        y_active_targets = np.zeros_like(active_indexes)
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
                ) if isinstance(n, int) else n[active_indexes[first_iter]]
                y_active_targets[first_iter] = np.argmax(active_targets[first_iter], axis=-1)

            iters += 1.

            gradient, output = self.adversarial_func([X_adversarial, y[active_indexes], active_targets, 0])

            y_thresh = output[range(len(output)), y_active_targets]
            y_best = np.argmax(output, axis=-1)
            completed = np.where((y_thresh > thresh) & (y_best == y_active_targets))[0]
            # extras = np.where((y_thresh >= y_output) | (extra_iters > 0))[0]
            if len(completed) > 0:
                pos = active_indexes[completed]
                diff = (inverse_transform(X_adversarial[completed]) - inverse_transform(X_active[completed])) / 255.
                amsd = self.compute_l2(diff)
                new_winner = np.where(amsd < scores['L2'][pos])[0]
                if len(new_winner) > 0:
                    ppos = pos[new_winner]
                    scores['L2'][ppos] = amsd[new_winner]
                    scores['AMSD_2'][ppos] = self.compute_amsd_2(diff[new_winner])
                    # X_best[completed[new_winner]] = X_adversarial[completed[new_winner]].copy()
                amsd = self.compute_l1(diff)
                new_winner = np.where(amsd < scores['L1'][pos])[0]
                if len(new_winner) > 0:
                    ppos = pos[new_winner]
                    scores['L1'][ppos] = amsd[new_winner]
                    scores['AMSD_1'][ppos] = self.compute_amsd_1(diff[new_winner])
                amsd = self.compute_l0(diff)
                new_winner = np.where(amsd < scores['L0'][pos])[0]
                if len(new_winner) > 0:
                    ppos = pos[new_winner]
                    scores['L0'][ppos] = amsd[new_winner]
                    scores['AMSD_0'][ppos] = self.compute_amsd_0(diff[new_winner])
                amsd = self.compute_l_inf(diff)
                new_winner = np.where(amsd < scores['L_inf'][pos])[0]
                if len(new_winner) > 0:
                    ppos = pos[new_winner]
                    scores['L_inf'][ppos] = amsd[new_winner]
                amsd = self.compute_amud(diff)
                new_winner = np.where(amsd < scores['amud'][pos])[0]
                if len(new_winner) > 0:
                    ppos = pos[new_winner]
                    scores['amud'][ppos] = amsd[new_winner]
                diff_image += np.sum(diff, axis=0)
                square_diff_image += np.sum(np.square(diff), axis=0)
                count_finished += len(completed)

            extras = np.where(((y_thresh > thresh) & (y_best == y_active_targets)) | (extra_iters > 0))[0]
            if len(extras) > 0:
                diff = X_adversarial[extras] - X_active[extras]
                sign = np.sign(diff)
                if l2 > 0.:
                    gradient[extras] += l2 * diff
                if l1 > 0.:
                    gradient[extras] += l1 * sign
                extra_iters[extras] += 1

            incompleted = np.where((extra_iters < extra_epochs) & ((extra_iters != 1) | (iters != 1)))[0]

            v_dX[incompleted] = beta1 * v_dX[incompleted] + (1. - beta1) * gradient[incompleted]
            s_dX[incompleted] = beta2 * s_dX[incompleted] + (1. - beta2) * np.square(gradient[incompleted])

            v_dX_c = v_dX[incompleted]  # / (1. - np.power(beta1, iters[incompleted]).reshape(ndims))
            s_dX_c = s_dX[incompleted]  # / (1. - np.power(beta2, iters[incompleted]).reshape(ndims))

            X_adversarial[incompleted] -= self.get_alpha(alpha, iters[incompleted]).reshape(ndims) * v_dX_c / (
                        np.sqrt(s_dX_c) + epsilon)  # / np.max(np.abs(gradient), axis=-1, keepdims=True)
            if constraint is not None:
                X_adversarial[incompleted] = constraint(X_adversarial[incompleted])

            if len(extras) > 0:
                new_sign = np.sign(X_adversarial[extras] - X_active[extras])
                X_adversarial[extras] = np.where(new_sign * sign <= 0, X_active[extras], X_adversarial[extras])

            completed = np.where((extra_iters >= extra_epochs) | ((extra_iters == 1) & (iters == 1)))[0]
            if len(completed) > 0:
                # if save_data is not None:
                #     if 'adversarial_data' in scores:
                #         scores['adversarial_data'][active_indexes[completed]] = X_best[completed].copy()
                #     else:
                #         urls = save_data[active_indexes[completed]].tolist()
                #         for url, x_b in zip(urls, X_best):
                #             np.save(url, x_b)
                nslots = min(len(inactive_indexes), len(completed))
                valid_pos = None
                if nslots < len(completed):
                    useless = completed[nslots:]
                    valid_pos = np.setdiff1d(np.arange(len(active_indexes), dtype=int), useless, assume_unique=True)
                    completed = completed[:nslots]
                if nslots:
                    active_indexes[completed] = inactive_indexes[:nslots]
                    active_targets[completed] = 0
                    y_active_targets[completed] = 0
                    v_dX[completed] = 0
                    s_dX[completed] = 0
                    iters[completed] = 0
                    extra_iters[completed] = 0
                    X_active[completed] = X[inactive_indexes[:nslots]]
                    X_min = min(X_min, X_active.min())
                    X_max = max(X_max, X_active.max())
                    X_adversarial[completed] = X_active[completed].copy()
                    X_best[completed] = X_active[completed].copy()
                    inactive_indexes = inactive_indexes[nslots:]
                if valid_pos is not None:
                    active_indexes = active_indexes[valid_pos]
                    active_targets = active_targets[valid_pos]
                    y_active_targets = y_active_targets[valid_pos]
                    v_dX = v_dX[valid_pos]
                    s_dX = s_dX[valid_pos]
                    iters = iters[valid_pos]
                    extra_iters = extra_iters[valid_pos]
                    X_active = X_active[valid_pos]
                    X_adversarial = X_adversarial[valid_pos]
                    X_best = X_best[valid_pos]
            if cont % 100 == 0:
                print('Cont : ', cont, ', Remaining : ', len(inactive_indexes) + len(active_indexes), ', Max iter : ',
                      iters.max(), ', Max extra : ', extra_iters.max(), ' Min extra : ', extra_iters.min(),
                      ' Min_thresh : ', y_thresh.min())
            cont += 1

        scores['L2'] = scores['L2'].tolist()
        scores['AMSD_2'] = scores['AMSD_2'].tolist()
        scores['AMSD_1'] = scores['AMSD_1'].tolist()
        scores['AMSD_0'] = scores['AMSD_0'].tolist()
        scores['amud'] = scores['amud'].tolist()
        scores['L1'] = scores['L1'].tolist()
        scores['L_inf'] = scores['L_inf'].tolist()
        scores['L0'] = scores['L0'].tolist()
        scores['min'] = X_min
        scores['max'] = X_max
        mean = (diff_image / count_finished)
        variance = square_diff_image / count_finished - np.square(mean)
        scores['image_mean'] = mean.tolist()
        scores['image_variance'] = variance.tolist()
        print('MEANS')
        print('AMSD_2: ', np.mean(scores['AMSD_2']))
        print('AMSD_1: ', np.mean(scores['AMSD_1']))
        print('AMSD_0: ', np.mean(scores['AMSD_0']))
        print('L2: ', np.mean(scores['L2']))
        print('L1: ', np.mean(scores['L1']))
        print('L0: ', np.mean(scores['L0']))
        print('L_inf: ', np.mean(scores['L_inf']))
        print('AMUD: ', np.mean(scores['amud']))
        print('MAX')
        print('AMSD_2: ', np.max(scores['AMSD_2']))
        print('AMSD_1: ', np.max(scores['AMSD_1']))
        print('AMSD_0: ', np.max(scores['AMSD_0']))
        print('L2: ', np.max(scores['L2']))
        print('L1: ', np.max(scores['L1']))
        print('L0: ', np.max(scores['L0']))
        print('L_inf: ', np.max(scores['L_inf']))
        print('AMUD: ', np.max(scores['amud']))
        print('MEDIAN')
        print('AMSD_2: ', np.median(scores['AMSD_2']))
        print('AMSD_1: ', np.median(scores['AMSD_1']))
        print('AMSD_0: ', np.median(scores['AMSD_0']))
        print('L2: ', np.median(scores['L2']))
        print('L1: ', np.median(scores['L1']))
        print('L0: ', np.median(scores['L0']))
        print('L_inf: ', np.median(scores['L_inf']))
        print('AMUD: ', np.median(scores['amud']))
        return scores

    # def gain_function(self, y_true, y_pred, y_target):
    #     y_pred_clipped = K.clip(y_pred, 0., 1.)
    #     return -1 * K.sum(y_target * K.log(y_pred_clipped), axis=-1)

    def gain_function(self, y_true, y_pred, y_target, threshold):
        if threshold < 0.5:
            y_target_max = K.max(y_target * y_pred, axis=-1, keepdims=True)
            return K.sum(K.relu((1 - y_target) * y_pred - y_target_max), axis=-1) + K.relu(threshold - y_target_max)
        else:
            return K.sum(K.relu(y_target * (threshold - y_pred)), axis=-1)

    def get_target(self, X, y_argmax, n, y_pred=None):
        if y_pred is None:
            y_pred = self.model.predict(X)
        y_pred_argsort = np.argsort(-1. * y_pred, axis=-1)
        y_targets = y_pred_argsort[y_pred_argsort != y_argmax.reshape((-1, 1))].reshape((y_pred_argsort.shape[0], -1))
        targets = np.zeros_like(y_pred)
        targets[range(len(targets)), y_targets[:, n - 1]] = 1.
        return targets
