from tensorflow.python.keras import Input
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Embedding, LSTM, Dense
from tensorflow.python.keras.preprocessing.text import Tokenizer

from loader import Loader


def build_model(use_gpu: bool = False, max_num_chars: int = None):
    """
    Builds the RNN-Model for character prediction.

    :param use_gpu: {bool} Uses Tensorflow GPU support if True, otherwise trains on CPU
    :return: Keras model
    """

    loader = Loader(1000, 0, 0)
    tokenizer = Tokenizer(
        filters='',
        split='°'
    )

    lengths = {}

    if max_num_chars is None:
        max_num_chars = 0
        for dataframe in loader:

            chars = set()

            for name in dataframe['name']:
                max_num_chars = max(max_num_chars, len(list(name)))
                lengths.setdefault(len(name), 0)
                lengths[len(name)] += 1

                chars.update(set(name))

            tokenizer.fit_on_texts(list(chars))

    tokenizer.fit_on_texts(['<pad>', '<end>'])
    input_layer = Input(shape=(1,))

    embedding_layer = Embedding(
        input_dim=max_num_chars,
        output_dim=len(tokenizer.word_index.keys())
    )

    lstm_layer = LSTM(
        units=32,
        return_sequences=False,
        return_state=False,
    )

    softmax_layer = Dense(
        units=len(tokenizer.word_index.keys()) + 1,
        activation='softmax'
    )

    softmax_layer = softmax_layer(
        inputs=lstm_layer(
            inputs=embedding_layer(
                inputs=input_layer
            )
        )
    )

    model = Model(input_layer, softmax_layer)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())

    return model, tokenizer, max_num_chars

