import pickle

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.utils import to_categorical

from build import build_model

import numpy as np

from loader import Loader

BATCH_SIZE = 200
NUM_BATCHES = 2
NUM_EPOCHS = 5

try:
    model = load_model('model.h5')

    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

except OSError:

    model, tokenizer, max_num_chars = build_model(max_num_chars=10)

    loader = Loader(
        batch_size=BATCH_SIZE,
        offset=0,
        limit=NUM_BATCHES * BATCH_SIZE
    )

    validation_loader = Loader(
        batch_size=BATCH_SIZE,
        offset=0,
        limit=NUM_BATCHES * BATCH_SIZE
    )

    gen = loader.get_train_generator(tokenizer, max_num_chars)

    model.fit_generator(
        generator=loader.get_train_generator(tokenizer, max_num_chars),
        steps_per_epoch=BATCH_SIZE * NUM_BATCHES,
        epochs=NUM_EPOCHS,
        validation_data=validation_loader.get_train_generator(tokenizer, max_num_chars),
        validation_steps=1
    )

model.save('model.h5')

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

sequence = list('Onion')
last_char = None
i = 0

while last_char != '<end>' and i < 20:
    tokenized_sequence = [tokenizer.word_index.get(char, tokenizer.word_index.get('pre')) for char in sequence]
    padded_sequence = pad_sequences([tokenized_sequence], 10)[0]
    padded_sequence = to_categorical(padded_sequence, num_classes=len(tokenizer.index_word) + 1)
    i = i + 1
    p = list(model.predict_classes(np.array(padded_sequence).reshape((1, 10, len(tokenizer.index_word) + 1))))
    last_char = tokenizer.index_word[p[0]]
    sequence += [tokenizer.index_word[p[0]]]
    print(sequence)
