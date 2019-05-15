from pandas import read_csv
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.utils import to_categorical


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

    def __create_dataframe_generator(self):
        """
        Create the data generator. This allows us to process huge amounts of data
        without actually loading it into memory all at once.

        :return: generator
        """

        def gen():
            i = 0
            for dataframe in read_csv('data/full/whats-on-the-menu/Dish.csv', delimiter=',', nrows=self.__limit if self.__limit > 0 else None,
                                      skiprows=self.__offset, chunksize=self.__batch_size):
                i += 1

                yield dataframe

        return gen()

    def __create_trainset_generator(self, tokenizer: Tokenizer, pad_to_length: int = 0, char_window_size: int = 10):

        def gen():

            i = 0

            while True:

                yielded = 0
                X = None
                Y = None
                batch_end = False

                for dataframe in read_csv('data/full/whats-on-the-menu/Dish.csv', delimiter=',', header=1, nrows=self.__limit if self.__limit > 0 else None,
                                          skiprows=self.__offset, chunksize=self.__batch_size):
                    i += 1

                    if batch_end:
                        break

                    for row in dataframe[dataframe.columns[1]]:

                        if self.__limit is not None and yielded >= self.__limit:
                            batch_end = True
                            break

                        windowed_tokenized_sequences = []

                        split_text = list(row)
                        if len(split_text) < char_window_size:
                            split_text = ['pre'] * (char_window_size - len(split_text)) + split_text

                        for window_start in range(0, len(split_text) - char_window_size + 2):

                            windowed_tokenized_sequences.append(
                                [tokenizer.word_index.get(char, tokenizer.word_index.get('pre')) for char in (split_text + ['<end>'])[window_start: window_start + char_window_size]]
                            )

                        windowed_tokenized_sequences = np.array(windowed_tokenized_sequences)
                        tokenized_char_phrases_X, tokenized_chars_y = windowed_tokenized_sequences[:, :-1], windowed_tokenized_sequences[:, -1]
                        padded_phrases_X = pad_sequences(tokenized_char_phrases_X, char_window_size)

                        one_hot_phrases = to_categorical(padded_phrases_X, num_classes=len(tokenizer.word_index) + 1)
                        one_hot_ys = to_categorical(tokenized_chars_y, num_classes=len(tokenizer.word_index) + 1)

                        if X is None:
                            X = one_hot_phrases
                            Y = one_hot_ys
                        else:
                            X = np.append(X, one_hot_phrases, axis=0)
                            Y = np.append(Y, one_hot_ys, axis=0)

                        if len(X) > self.__batch_size:
                            yielded += self.__batch_size
                            yield X[:self.__batch_size], Y[:self.__batch_size]

                            X = X[self.__batch_size:]
                            Y = Y[self.__batch_size:]

        return gen()

    def get_generator(self):
        return self.__create_dataframe_generator()

    def get_train_generator(self, tokenizer: Tokenizer, pad_to_length: int = 0):
        return self.__create_trainset_generator(tokenizer, pad_to_length)
