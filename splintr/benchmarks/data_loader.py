import os
import pickle
import tarfile
import requests
import numpy as np


""" Returns Generators For Training And Testing CIFAR-100 Data. 
    Generators Produce Tuples Of Image (32, 32, 3) And One-Hot Encoded
    Labels. """

def cifar100():

    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    open('cifar-100.tar.gz', "wb").write(requests.get(url).content)

    tar_out = tarfile.open('cifar-100.tar.gz', 'r:gz')

    for member in tar_out.getmembers():
        if (member.name == 'cifar-100-python/train'):
            train = pickle.load(tar_out.extractfile(member), encoding = 'bytes')

        elif (member.name == 'cifar-100-python/test'):
            test = pickle.load(tar_out.extractfile(member), encoding = 'bytes')

    def data_generator(data):
        for i in range(len(data[b'data'])):
            image = data[b'data'][i]
            label = data[b'fine_labels'][i]

            image = image.reshape((3, 1024))
            out_image = []

            for j in range(32):
                row = []

                for k in range(32):
                    row.append([image[0][j * 32 + k], image[1][j * 32 + k], image[2][j * 32 + k]])

                out_image.append(row)

            yield (np.array(out_image), label)

    os.remove('cifar-100.tar.gz')

    return (data_generator(train), data_generator(test))
