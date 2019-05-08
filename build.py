from tensorflow.python.keras.layers import Embedding

from loader import Loader


def build_model(use_gpu: bool = False, max_num_chars: int = None):
    """
    Builds the RNN-Model for character prediction.

    :param use_gpu: {bool} Uses Tensorflow GPU support if True, otherwise trains on CPU
    :return: Keras model
    """

    loader = Loader(1000, 0, 0)

    if max_num_chars is None:
        max_num_chars = 0
        for dataframe in loader:
            for name in dataframe['name']:
                max_num_chars = max(max_num_chars, len(name))

    embedding_layer = Embedding(
        input_dim=max_num_chars,
        output_dim=10
    )

    print('MAX----')
    print(max_num_chars)


build_model()