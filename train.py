import argparse

from timeit import default_timer as timer
import numpy as np
import pickle
import random
import string
import sys

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.utils import to_categorical

from build import build_model
from loader2 import Loader

INITIAL_SEQUENCES = (
    'GVoRUcZybxnbknW',
    'TKWLvaUtsiBvjPm',
    'fewKeuFYJecyCAk',
    'KXntyJsfzrZFAJT',
    'jTNkWHsSswfEqhE',
    'uNjcueOEZadlCMi',
    'XESvYRSRqVSVNQy',
    'EcWjyUUvFZlzIVr',
    'A',
    'B',
    'C',
    'Chicken',
    'cWjyUUvFZlzIVaC',
    'XESvYRSRqVSVN B'
)

START_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ'

parser = argparse.ArgumentParser()
parser.add_argument('--configuration', default=None)
parser.add_argument('--force_train', default=False)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_batches', default=1, type=int)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--num_units', default=64, type=int)
parser.add_argument('--window_size', default=10, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout_rate', default=0, type=float)
parser.add_argument('--use_gpu', default=False)
args = parser.parse_args()

model = None
tokenizer = None

class EpochDone(Callback):

    start_time = None
    sum_times = 0

    def on_epoch_begin(self, epoch, logs=None):

        print('epoch begin')
        self.start_time = timer()

    def on_epoch_end(self, epoch, logs=None):

        if self.start_time is not None:
            took = timer() - self.start_time
            print('Epoch took {} '.format(took))
            self.sum_times += took

    def on_train_end(self, logs=None):
        print('Total time: ', self.sum_times)
        print('Average: ', self.sum_times/10)


if args.configuration:
    try:
        model = load_model('checkpoints/checkpoint_{}.best_val_acc.hdf5'.format(args.configuration), )

        with open('tokenizers/tokenizer_{}.pickle'.format(args.configuration), 'rb') as handle:
            tokenizer = pickle.load(handle)

    except OSError:

        sys.stderr.write('Configuration {} not found!'.format(args.configuration))
        model = None
        exit(1)

if args.configuration is None or args.force_train:

    args.configuration = '{}_{}_{}_{}_{}_{}_{}'.format(
        args.batch_size,
        args.num_batches,
        args.num_epochs,
        args.num_units,
        args.num_layers,
        args.dropout_rate,
        args.window_size,
    )

    if model is None or tokenizer is None or args.force_train:
        model, tokenizer = build_model(
            use_gpu=args.use_gpu,
            num_units=args.num_units,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            num_layers=args.num_layers,
            window_size=args.window_size,
        )
    else:
        print('MODEL LOADED FROM DISK')

    with open('tokenizers/tokenizer_{}.pickle'.format(args.configuration), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loader = Loader(
        batch_size=args.batch_size,
        offset=0,
        num_batches=args.num_batches,
        epochs=args.num_epochs
    )

    validation_loader = Loader(
        batch_size=args.batch_size,
        offset=211019,
        num_batches=args.num_batches,
        epochs=args.num_epochs
    )

    model.fit_generator(
        generator=loader.get_train_generator(tokenizer, args.window_size),
        steps_per_epoch=args.batch_size * args.num_batches,
        epochs=args.num_epochs,
        validation_data=validation_loader.get_train_generator(tokenizer, args.window_size),
        validation_steps=32,
        callbacks=[
            ModelCheckpoint(
                'checkpoints/checkpoint_{}.best_val_acc.hdf5'.format(args.configuration),
                monitor='val_accuracy',
                verbose=2,
                save_best_only=True,
                mode='max',
            ),
            # EarlyStopping(
            #     monitor='accuracy',
            #     min_delta=0.001,
            #     patience=5,
            #     verbose=2
            # ),
            EpochDone()
        ]
    )

num_samples = 0

for sequence in INITIAL_SEQUENCES:
# for sequence in range(0, 30):

    i = 0
    num_samples += 1
    # sequence = [random.choice(string.ascii_letters) for r in range(0, args.window_size - 1)] + [START_LETTERS[random.randint(0, len(START_LETTERS)-1)]]
    sequence = list(sequence)
    if len(sequence) < args.window_size:
        sequence = ['pad'] * (args.window_size - len(sequence)) + sequence

    initial_sequence = sequence.copy()
    out_sequence = []
    last_char = None

    while last_char != '<end>' and i < 60:
        tokenized_sequence = [tokenizer.word_index.get(char, tokenizer.word_index.get('pre')) for char in sequence]
        padded_sequence = pad_sequences([tokenized_sequence], args.window_size)[0]
        padded_sequence = to_categorical(padded_sequence, num_classes=len(tokenizer.index_word) + 1)
        i = i + 1
        p = list(model.predict_classes(np.array(padded_sequence).reshape((1, args.window_size, len(tokenizer.index_word) + 1))))
        # print(p)
        last_char = tokenizer.index_word[p[0]]
        sequence += [tokenizer.index_word[p[0]]]
        out_sequence += [tokenizer.index_word[p[0]]]

    print('{} -> {}'.format(
        ''.join(initial_sequence),
        '{}{}'.format(initial_sequence[-1], ''.join(out_sequence[:-1])))
    )
