from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, CuDNNLSTM
from tensorflow.python.keras.preprocessing.text import Tokenizer

from loader import Loader


def build_model(use_gpu: bool = False, max_num_chars: int = None, num_units: int = 64):
    """
    Builds the RNN-Model for character prediction.

    :param use_gpu: {bool} Uses Tensorflow GPU support if True, otherwise trains on CPU
    :return: Keras model
    """

    loader = Loader(1000, 0, 0)
    tokenizer = Tokenizer(
        filters='',
        split='°',
        lower=False
    )

    lengths = {}

    update_max_num_chars = max_num_chars is None
    if update_max_num_chars is None:
        max_num_chars = 0

    for dataframe in loader:

        chars = set()

        for name in dataframe['name']:
            if len(name) > 200:
                continue

            if update_max_num_chars:
                max_num_chars = max(max_num_chars, len(list(name)))

            lengths.setdefault(len(name), 0)
            lengths[len(name)] += 1

            chars.update(set(name))

        tokenizer.fit_on_texts(list(chars))

    tokenizer.fit_on_texts(['pre', '<end>'])

    model = Sequential()
    model.add((CuDNNLSTM if use_gpu else LSTM)(num_units, input_shape=(max_num_chars, len(tokenizer.index_word) + 1)))
    model.add(Dense(len(tokenizer.index_word) + 1, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())

    return model, tokenizer, max_num_chars

