import numpy as np


class AdversarialGenerator:

    def __init__(self, model, loss_generator, optimizer, transformer, dists):
        self.model = model
        self.loss_generator = loss_generator
        self.optimizer = optimizer
        self.transformer = transformer
        self.dists = dists

    def get_adversarial_scores(
            self, X, y, batch_size=10, extra_epochs=1000
    ):
        # hechas pruebas con l2 = 5e-2
        info = {'scores': {}}
        for dist in self.dists:
            info['scores'][dist.__name__] = np.Inf * np.ones(len(X))

        active_indexes = np.arange(min(len(X), batch_size)).astype(int)
        inactive_indexes = np.arange(min(len(X), batch_size), len(X)).astype(int)

        X_active = X[active_indexes] if self.transformer is None else self.transformer.convert_to_model_input(X[active_indexes])
        X_adversarial = X_active.copy()

        iters = np.zeros(batch_size)
        extra_iters = np.zeros(batch_size)

        ndims = [1] * X_active.ndim
        ndims[0] = -1

        cont = 0
        output = np.zeros((len(active_indexes), ) + self.model.output_shape[1:])
        while len(active_indexes):
            output[iters == 0] = self.model.predict(X_adversarial[iters == 0])

            active_targets = self.loss_generator.generate_targets(y[active_indexes], output)
            iters += 1.

            gradient, output, loss = self.loss_generator.get_gradient_and_output(X_adversarial, y[active_indexes], active_targets, extra_iters)

            completed = np.where(loss == 0)[0]
            # extras = np.where((y_thresh >= y_output) | (extra_iters > 0))[0]
            if len(completed) > 0:
                pos = active_indexes[completed]
                diff = X_adversarial[completed] - X_active[completed] if self.transformer is None else \
                       self.transformer.normalize(self.transformer.recover_from_model_input(X_adversarial[completed] - X_active[completed]))
                for dist in self.dists:
                    score = dist(diff)
                    new_winner = np.where(score < info['scores'][dist.__name__][pos])[0]
                    if len(new_winner) > 0:
                        winner_pos = pos[new_winner]
                        info['scores'][dist.__name__][winner_pos] = score[new_winner]

            gradient = self.optimizer.step(gradient)
            sign = np.sign(X_adversarial - X_active)
            X_adversarial -= gradient

            incompleted = np.where((extra_iters < extra_epochs) & ((extra_iters != 1) | (iters != 1)))[0]

            if self.transformer is not None:
                X_adversarial[incompleted] = self.transformer.clip(X_adversarial[incompleted])

            extras = np.where(extra_iters > 0)[0]

            if len(extras) > 0.:
                new_sign = np.sign(X_adversarial - X_active)
                X_adversarial = np.where(new_sign * sign <= 0, X_active, X_adversarial)

            completed = np.where((extra_iters >= extra_epochs) | ((extra_iters == 1) & (iters == 1)))[0]
            if len(completed) > 0:
                nslots = min(len(inactive_indexes), len(completed))
                valid_pos = None
                if nslots < len(completed):
                    useless = completed[nslots:]
                    valid_pos = np.setdiff1d(np.arange(len(active_indexes), dtype=int), useless, assume_unique=True)
                    completed = completed[:nslots]
                if nslots:
                    active_indexes[completed] = inactive_indexes[:nslots]
                    active_targets[completed] = 0
                    v_dX[completed] = 0
                    s_dX[completed] = 0
                    iters[completed] = 0
                    extra_iters[completed] = 0
                    hits[completed] = 0
                    X_active[completed] = X[inactive_indexes[:nslots]]
                    X_min = min(X_min, X_active.min())
                    X_max = max(X_max, X_active.max())
                    X_adversarial[completed] = X_active[completed].copy()
                    X_best[completed] = X_active[completed].copy()
                    inactive_indexes = inactive_indexes[nslots:]
                if valid_pos is not None:
                    active_indexes = active_indexes[valid_pos]
                    active_targets = active_targets[valid_pos]
                    v_dX = v_dX[valid_pos]
                    s_dX = s_dX[valid_pos]
                    iters = iters[valid_pos]
                    extra_iters = extra_iters[valid_pos]
                    hits = hits[valid_pos]
                    X_active = X_active[valid_pos]
                    X_adversarial = X_adversarial[valid_pos]
                    X_best = X_best[valid_pos]

            if cont % 100 == 0 and len(active_indexes) > 0:
                print('Cont : ', cont, ', Remaining : ', len(inactive_indexes) + len(active_indexes),
                      ', Max iter : ', iters.max(), ', Max extra : ', extra_iters.max(), ', Min extra : ', extra_iters.min(),
                      ', Max_diff : ', (y_thresh - y_output).max(), ', Min diff : ', (y_thresh - y_output).min())
            cont += 1
        scores['L2'] = (scores['L2'] / (X_max - X_min)).tolist()
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
        print('L2: ', np.mean(scores['L2']))
        print('L1: ', np.mean(scores['L1']))
        print('L0: ', np.mean(scores['L0']))
        print('L_inf: ', np.mean(scores['L_inf']))
        print('AMUD: ', np.mean(scores['amud']))
        return scores




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
