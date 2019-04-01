from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import numpy as np
import threading
from keras.preprocessing import image


class NoiseDataGenerator(ImageDataGenerator):
    # Class is a dataset wrapper for better training performance
    def __init__(self, negative=False,
                 gaussian_stddev=0.,
                 uniform_range=0.,
                 minval=None,
                 maxval=None,
                 **kwargs):
        self.gaussian_stddev = gaussian_stddev
        self.uniform_range = uniform_range
        self.minval = minval
        self.maxval = maxval
        self.negative = negative
        super(NoiseDataGenerator, self).__init__(**kwargs)

    def random_transform(self, x, seed=None):
        x = super(NoiseDataGenerator, self).random_transform(x, seed)

        if self.gaussian_stddev:
            x += np.random.randn(*x.shape) * self.gaussian_stddev

        if self.uniform_range:
            x += np.random.uniform(-self.uniform_range, self.uniform_range)

        if self.minval:
            x = np.maximum(self.minval, x)

        if self.maxval:
            x = np.minimum(self.maxval, x)

        if self.negative:
            if np.random.random() < 0.5:
                maxval = 1. if self.maxval is None else self.maxval
                x = maxval - x
        return x


class FromDiskGenerator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(
            self, x_set, target_size=None, preprocess_func=None, batch_size=256,
    ):
        self.x = x_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x)).astype(int)
        self.target_size = target_size
        self.preprocess_func = preprocess_func
        self.memory_batch = {}
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idxs):
        if isinstance(idxs, slice):
            idxs = np.arange(idxs.start, idxs.stop, idxs.step)
        for idx in idxs:
            if idx not in self.memory_batch:
                self.lock.acquire()
                memory_batch_idx = idx // self.batch_size
                self.__load_batch_data(memory_batch_idx)
                self.lock.release()
        target_size = self.memory_batch[idxs[0]].shape
        output = np.zeros(((len(idxs), ) + target_size))
        for i, idx in enumerate(idxs):
            output[i] = self.memory_batch[idx]
            del self.memory_batch[idx]
        return output

    def __load_batch_data(self, memory_batch_idx):
        inds = self.indices[
            memory_batch_idx * self.batch_size:min(len(self.x), (memory_batch_idx + 1) * self.batch_size)
        ]
        memory_batch_x = np.array([image.img_to_array(image.load_img(self.x[ind], target_size=self.target_size[:2], interpolation='bicubic')) for ind in inds])
        if self.preprocess_func is not None:
            memory_batch_x = self.preprocess_func(memory_batch_x)
        memory_batch_dict = dict(zip(inds, memory_batch_x))
        self.memory_batch = {**self.memory_batch, **memory_batch_dict}
