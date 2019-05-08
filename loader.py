from pandas import read_csv


class Loader(object):

    __batch_size: int = 1000
    __offset: int = 0
    __limit: int = 0
    __generator = None

    def __init__(self, batch_size: int = 1000, offset: int = 0, limit: int = 0):
        self.__batch_size = batch_size
        self.__offset = offset
        self.__limit = limit
        self.__generator = self.__create_generator()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.__generator)

    def __create_generator(self):
        """
        Create the data generator. This allows us to process huge amounts of data
        without actually loading it into memory all at once.

        :return: generator
        """

        def gen():

            print('yep')

            i = 0
            for dataframe in read_csv('data/full/whats-on-the-menu/Dish.csv', delimiter=',', nrows=self.__limit if self.__limit > 0 else None, skiprows=self.__offset, chunksize=self.__batch_size):
                print(i)
                i += 1
                yield dataframe

        return gen()
