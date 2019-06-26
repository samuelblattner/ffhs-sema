from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, CuDNNLSTM
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.layers.core import Dropout

from loader import Loader


def build_model(
        use_gpu: bool = False,
        num_units: int = 64,
        num_layers: int = 1,
        dropout_rate: float = 0.0,
        batch_size: int = 1000,
        window_size: int = 10,
        num_params: int = 0):
    """
    Builds the RNN-Model for character prediction.

    :param window_size: Sequence size
    :param batch_size: {int} Size of batch
    :param dropout_rate: {float} Regulating Dropout rate between layers
    :param num_layers: {int} Number of layers to build
    :param num_units: {int} Number of LSTM-Units to use in network
    :param use_gpu: {bool} Uses Tensorflow GPU support if True, otherwise trains on CPU
    :param num_params: {int} Number of control parameters
    :return: Keras model
    """

    # Load max 5000 entries from the dataset to build the Tokenizer / vocabulary
    loader = Loader(min(batch_size, 5000), 0)
    tokenizer = Tokenizer(
        filters='',
        split='°',
        lower=False
    )

    for dataframe in loader:

        chars = set()

        for name in dataframe['name']:
            chars.update(set(str(name)))

        tokenizer.fit_on_texts(list(chars))

    tokenizer.fit_on_texts(['pre', '<end>', 'pad'])

    # Build Keras Model
    model = Sequential()
    for r in range(0, max(num_layers - 1, 0)):
        model.add(
            layer=(CuDNNLSTM if use_gpu else LSTM)(
                num_units,
                input_shape=(window_size, len(tokenizer.index_word) + 1 + num_params),
                return_sequences=True
            )
        )
        model.add(Dropout(dropout_rate))

    model.add(
        layer=(CuDNNLSTM if use_gpu else LSTM)(
            num_units,
            input_shape=(window_size, len(tokenizer.index_word) + 1 + num_params)
        )
    )
    model.add(Dense(len(tokenizer.index_word) + 1, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Show summary
    print(model.summary())

    return model, tokenizer
