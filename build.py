from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, CuDNNLSTM, SimpleRNNCell
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.layers.core import Dropout

from loader import Loader


def build_model(use_gpu: bool = False, num_units: int = 64, num_layers: int = 1, dropout_rate=0.0, batch_size: int = 1000, num_batches: int = 1, window_size: int = 10):
    """
    Builds the RNN-Model for character prediction.

    :param use_gpu: {bool} Uses Tensorflow GPU support if True, otherwise trains on CPU
    :return: Keras model
    """

    loader = Loader(min(batch_size, 5000), 0, num_batches=num_batches, limit=5000)
    tokenizer = Tokenizer(
        filters='',
        split='°',
        lower=False
    )

    for dataframe in loader:

        chars = set()

        for name in dataframe['name']:
            if len(str(name)) > 200:
                continue

            chars.update(set(str(name)))

        tokenizer.fit_on_texts(list(chars))

    tokenizer.fit_on_texts(['pre', '<end>', 'pad'])

    model = Sequential()
    for layer in range(0, max(num_layers - 1, 0)):
        model.add((CuDNNLSTM if use_gpu else LSTM)(num_units,  input_shape=(window_size, len(tokenizer.index_word) + 1 + 1,), return_sequences=True))
        model.add(Dropout(dropout_rate))

    model.add((CuDNNLSTM if use_gpu else LSTM)(num_units,  input_shape=(window_size, len(tokenizer.index_word) + 1 + 1)))
    model.add(Dense(len(tokenizer.index_word) + 1, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())

    return model, tokenizer
