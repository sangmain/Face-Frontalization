import numpy as np
from skimage.io import imread
import keras
from glob import glob
import PIL.Image as pilimg

class DataGenerator(keras.utils.Sequence):
    def __init__(self, sideslist, frontslist, batch_size=32, dim = (128, 128), n_channels = 3, shuffle = True):
        self.dim = dim
        self.batch_size = batch_size
        self.sideslist = sideslist
        self.frontslist = frontslist
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sideslist) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        sideslist_temp = [self.sideslist[k] for k in indexes]
        frontslist_temp = [self.frontslist[k] for k in indexes]

        sides, fronts = self.__data_generation(sideslist_temp, frontslist_temp)

        return sides, fronts

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sideslist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sideslist, frontslist):
        X = np.empty((self.batch_size, * self.dim, self.n_channels))
        Y = np.empty((self.batch_size, * self.dim, self.n_channels))
        i = 0

        for sidename, frontname in zip(sideslist, frontslist):
            side = pilimg.open(sidename)
            side = np.array(side)
            X[i] = side
            front = pilimg.open(frontname)
            front = np.array(front)
            Y[i] = front
            i += 1


        return self.preprossing(X), self.preprossing(Y)

    def preprossing(self, img):
        return (img / 255 * 2 - 1)
        # return (img / 127.5 - 1)

class DataGenerator_predict(keras.utils.Sequence):
    def __init__(self, sideslist, batch_size = 32, dim = (128, 128), n_channels = 3, shuffle = True):
        self.dim = dim
        self.batch_size = batch_size
        self.sideslist = sideslist
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sideslist) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        sideslist_temp = [self.sideslist[k] for k in indexes]

        sides = self.__data_generation(sideslist_temp)

        return sides

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sideslist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sideslist):
        X = np.empty((self.batch_size, * self.dim, self.n_channels))
        i = 0

        for sidename in sideslist:
            side = pilimg.open(sidename)
            side = np.array(side)
            X[i] = side
            i += 1

        return self.preprossing(X)
    
    def preprossing(self, img):
        return (img / 255 * 2 - 1)
        # return (img / 127.5 - 1)