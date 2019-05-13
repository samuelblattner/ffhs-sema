from typing import List

from pandas import read_csv
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np

class Loader(object):

    __batch_size: int = 1000
    __offset: int = 0
    __limit: int = 0
    __generator = None

    def __init__(self, batch_size: int = 1000, offset: int = 0, limit: int = 0):
        self.__batch_size = batch_size
        self.__offset = offset
        self.__limit = limit
        self.__generator = self.__create_dataframe_generator()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.__generator)

    def __pad_to_length(self, sequence: List, length):
        sequence += ['<pad>'] * (length - len(sequence))
        return sequence

    def __create_dataframe_generator(self):
        """
        Create the data generator. This allows us to process huge amounts of data
        without actually loading it into memory all at once.

        :return: generator
        """

        def gen():

            i = 0
            for dataframe in read_csv('data/full/whats-on-the-menu/Dish.csv', delimiter=',', nrows=self.__limit if self.__limit > 0 else None, skiprows=self.__offset, chunksize=self.__batch_size):
                i += 1

                yield dataframe

        return gen()

    def __create_trainset_generator(self, tokenizer: Tokenizer, pad_to_length: int = 0):

        def gen():

            i = 0

            print(self.__limit)
            print(self.__batch_size)

            for dataframe in read_csv('data/full/whats-on-the-menu/Dish.csv', delimiter=',', header=1, nrows=self.__limit if self.__limit > 0 else None, skiprows=self.__offset, chunksize=self.__batch_size):
                i += 1

                print('CHUNK START WITH ')
                print(len(dataframe))

                for row in dataframe[dataframe.columns[1]]:

                    x = np.array(tokenizer.texts_to_sequences(self.__pad_to_length(list(row), pad_to_length))).reshape((pad_to_length, 1))
                    y = np.array(tokenizer.texts_to_sequences(self.__pad_to_length(list(row)[1:] + ['<end>'], pad_to_length))).reshape((pad_to_length, 1))

                    yield x, y

                print('CHUNCK END =============')
                print(i)

        return gen()

    def get_generator(self):
        return self.__create_dataframe_generator()

    def get_train_generator(self, tokenizer: Tokenizer, pad_to_length: int = 0):
        return self.__create_trainset_generator(tokenizer, pad_to_length)
