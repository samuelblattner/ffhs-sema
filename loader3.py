import string

from pandas import read_csv
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.utils import to_categorical

MEAT_LIST = (

    # Chicken
    'chicken',
    'hühnchen',
    'hähnchen',
    'poulet',
    'pollo',
    'kip'
    
    # Duck
    'duck',
    'ente',
    'canard',
    'cane',
    'anatra',
    'eend',
    'pata',
    'pato'
    
    # Beef
    'beef',
    'rind',
    'steak',
    'boeuf',
    'boeuf',
    'manzo',
    'carne de res',
    'rundvlees'
    
    # Veal
    'veal',
    'kalb',
    'veau',
    'vitello',
    'ternera',
    'kalfsvlees'
    
    # Venison
    'venison',
    'wild',
    'venaison'
    'carne di cervo'
    'venado'
    'hertevlees'
    
    # Lamb
    'lamb',
    'lamm',
    'agneau'
    'agnello',
    'cordero'
    'cordera'
    'lam'
    
    # Pork
    'pork',
    'porc',
    'maiale',
    'cerda',
    'cerdo',
    'varkensvlees',
)


class Loader(object):
    __batch_size: int = 1000
    __offset: int = 0
    __limit: int = 0
    __epochs: int = 1
    __window_size: int = 10
    __generator = None

    __num_lines_yielded: int = 0
    __total_samples_yielded: int = 0

    def __init__(self, batch_size: int = 1000, offset: int = 0, limit: int = 0, epochs: int = 1, window_size: int = 10, num_batches: int = 1):
        self.__batch_size = batch_size
        self.__num_batches = num_batches
        self.__offset = offset
        self.__limit = limit
        self.__epochs = epochs
        self.__window_size = window_size
        self.__generator = self.__create_dataframe_generator()
        self.__num_lines_yielded = 0
        self.__total_samples_yielded = 0

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

    def __create_trainset_generator(self, tokenizer: Tokenizer, char_window_size: int = 10):

        def gen(loader):

            initial_offset_lines_skipped = 0

            while True:

                batches_yielded = 0
                samples_yielded = 0
                self.__total_samples_yielded = 0
                X = None
                Y = None

                for dataframe in read_csv(
                        filepath_or_buffer='data/full/whats-on-the-menu/Dish.csv',
                        delimiter=',',
                        header=0,
                        chunksize=1000):

                    for i, row in dataframe.iterrows():

                        name = row[1]
                        year = ((float(row[5]) - 1800) / 300)

                        for word in name.strip(string.punctuation).strip().split(' '):
                            if word.lower() in MEAT_LIST:
                                vegetarian = False
                                break
                        else:
                            vegetarian = True

                        if self.__offset > 0 and initial_offset_lines_skipped < self.__offset:
                            initial_offset_lines_skipped += 1
                            continue

                        # vegetarian = True

                        windowed_tokenized_sequences = []

                        # Split menu text up into characters and attach stop marker
                        split_text = list(name)
                        split_text_with_end = split_text + ['<end>']

                        # Generate windows
                        for window_end in range(0, len(split_text_with_end)):

                            padding_length = char_window_size - window_end

                            windowed_tokenized_sequences.append([
                                    tokenizer.word_index.get(char, tokenizer.word_index.get('pre')) for char in (
                                        ['pad'] * max(0, padding_length) + split_text_with_end[max(0, window_end-char_window_size):window_end + 1])
                                ])

                        # print(windowed_tokenized_sequences)

                        windowed_tokenized_sequences = np.array(windowed_tokenized_sequences)
                        # print(tokenizer.sequences_to_texts(windowed_tokenized_sequences[:, :-1]))
                        # print(tokenizer.sequences_to_texts([windowed_tokenized_sequences[:, -1]]))
                        tokenized_char_phrases_X, tokenized_chars_y = windowed_tokenized_sequences[:, :-1], windowed_tokenized_sequences[:, -1]
                        padded_phrases_X = pad_sequences(tokenized_char_phrases_X, char_window_size)

                        # print(padded_phrases_X.shape)
                        # print(np.array([[[year]] * char_window_size] * len(padded_phrases_X)).shape)
                        # print(to_categorical(padded_phrases_X, num_classes=len(tokenizer.index_word) + 1).shape)
                        one_hot_phrases = np.append(
                            to_categorical(padded_phrases_X, num_classes=len(tokenizer.index_word) + 1),
                            np.array([[[year]] * char_window_size] * len(padded_phrases_X)),
                            axis=2)

                        one_hot_phrases = np.append(
                            one_hot_phrases,
                            np.array([[[1.0 if vegetarian else 0.0]] * char_window_size] * len(padded_phrases_X)),
                            axis=2)

                        # print(one_hot_phrases)

                        # print(padded_phrases_X)
                        # print(one_hot_phrases)
                        one_hot_ys = to_categorical(tokenized_chars_y, num_classes=len(tokenizer.index_word) + 1)

                        if X is None:
                            X = one_hot_phrases
                            Y = one_hot_ys
                        else:
                            X = np.append(X, one_hot_phrases, axis=0)
                            Y = np.append(Y, one_hot_ys, axis=0)

                        loader.__num_lines_yielded += 1
                        samples_yielded += len(windowed_tokenized_sequences)

                        # print(samples_yielded)
                        if samples_yielded >= self.__batch_size:
                            # print(X.shape)

                            yield X[:self.__batch_size], Y[:self.__batch_size]

                            batches_yielded += 1

                            X = X[self.__batch_size:]
                            Y = Y[self.__batch_size:]

                            self.__total_samples_yielded += samples_yielded

                            samples_yielded = len(X)

        return gen(self)

    def get_num_lines_yielded(self):
        return self.__num_lines_yielded

    def get_generator(self):
        return self.__create_dataframe_generator()

    def get_train_generator(self, tokenizer: Tokenizer, pad_to_length: int = 0):
        return self.__create_trainset_generator(tokenizer, pad_to_length)

    def get_num_lines(self):
        lines = 0
        for dataframe in read_csv('data/full/whats-on-the-menu/Dish.csv', delimiter=',', nrows=self.__limit if self.__limit > 0 else None,
                                  skiprows=self.__offset, chunksize=10000):
            lines += len(dataframe)

        return lines

    def get_total_samples_yielded(self):
        return self.__total_samples_yielded
