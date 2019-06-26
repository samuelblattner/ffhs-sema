import argparse
import random
import string


from timeit import default_timer as timer
import numpy as np
import pickle
import sys
from matplotlib import pyplot as plt

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.utils import to_categorical

from build import build_model
from loader3 import Loader

INITIAL_SEQUENCES = (
    'A',
    'E',
    'I',
    'O',
    'U',
    'C',
    'S',
    'B',
    'T',
    # 'Chicken',
    # 'Steak',
    # 'Burger',
    # 'Tomato'
)

# INITIAL_SEQUENCES = (
#     'Chicken',
#     'Chicken',
#     'Chicken',
#     'Chicken',
#     'Chicken',
#     'Chicken',
#     'Chicken',
#     'Chicken',
#     'Chicken',
#     'Chicken',
# )

START_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ'

parser = argparse.ArgumentParser()
parser.add_argument('--configuration', default=None)
parser.add_argument('--force_train', default=False)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_batches', default=1, type=int)
parser.add_argument('--perc_lines', default=0, type=float)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--num_units', default=64, type=int)
parser.add_argument('--window_size', default=10, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout_rate', default=0, type=float)
parser.add_argument('--use_gpu', default=False)
parser.add_argument('--use_year', default=False)
args = parser.parse_args()

model = None
tokenizer = None

class LinesYielded(Callback):

    generator = None

    def __init__(self, loader):
        self.generator = loader
        super(LinesYielded, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        num = 0
        num_samples = 0
        if self.generator:
            num = self.generator.get_num_lines_yielded()
            num_samples = self.generator.get_total_samples_yielded()

        print('Number of lines yielded: ', num)
        print('Number of samples yielded: ', num_samples)


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


if args.perc_lines > 0.0:

    loader = Loader()
    args.num_batches = int((args.perc_lines * loader.get_num_lines() * 26)/(args.batch_size * args.num_epochs))
    print('To see {} of dataset, {} batches with {} training steps are run for {} epochs.'.format(
        args.perc_lines,
        args.num_batches,
        args.batch_size,
        args.num_epochs
    ))

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

    args.configuration = 'VEG_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.perc_lines,
        args.batch_size,
        args.num_batches,
        args.num_epochs,
        args.num_units,
        args.num_layers,
        args.dropout_rate,
        args.window_size,
        'y' if args.use_year else 'ny'
    )

    if model is None or tokenizer is None:
        model, tokenizer = build_model(
            use_gpu=args.use_gpu,
            num_units=args.num_units,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            num_layers=args.num_layers,
            window_size=args.window_size,
            dropout_rate=args.dropout_rate
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

    train_history = model.fit_generator(
        generator=loader.get_train_generator(tokenizer, args.window_size),
        steps_per_epoch=args.num_batches,
        epochs=args.num_epochs,
        validation_data=validation_loader.get_train_generator(tokenizer, args.window_size),
        validation_steps=26 * 20,
        callbacks=[
            ModelCheckpoint(
                'checkpoints/checkpoint_{}.best_val_acc.hdf5'.format(args.configuration),
                monitor='val_accuracy',
                verbose=2,
                save_best_only=True,
                mode='max',
            ),
            EarlyStopping(
                monitor='accuracy',
                min_delta=0.001,
                patience=25,
                verbose=2
            ),
            EpochDone(),
            LinesYielded(loader)
        ]
    )

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    acc = train_history.history['accuracy']
    val_acc = train_history.history['val_accuracy']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.axis([0, args.num_epochs, 0, 3.5])
    plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])
    plt.title(args.configuration)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()

    fig1.savefig('train_histories/{}.pdf'.format(args.configuration), format='pdf')

num_samples = 0

words = [''] + [i for w,i in tokenizer.index_word.items()]

year = ((float(2000) - 1800) / 300) if args.use_year else None
vegetarian = False

print(year)
out = []
for r in range(0, 112):
    for sequence in INITIAL_SEQUENCES:
    # for sequence in range(0, 30):
    # for sequence in range(0, 1000):

        i = 0
        num_samples += 1
        # sequence = [] + [START_LETTERS[random.randint(0, len(START_LETTERS)-1)]]
        sequence = list(sequence)
        # sequence = list('Chicken')
        initial_sequence = sequence.copy()
        if len(sequence) < args.window_size:
            sequence = ['pad'] * (args.window_size - len(sequence)) + sequence

        out_sequence = []
        last_char = None

        while last_char != '<end>' and i < 500:
            tokenized_sequence = [tokenizer.word_index.get(char, tokenizer.word_index.get('pre')) for char in sequence]
            padded_sequence = pad_sequences([tokenized_sequence], args.window_size)[0]

            if year:
                padded_sequence = np.append(
                    to_categorical(padded_sequence, num_classes=len(tokenizer.index_word) + 1),
                    np.array([[year]] * args.window_size),
                    axis=1)

                padded_sequence = np.append(
                    padded_sequence,
                    np.array([[1.0 if vegetarian else 0.0 ]] * args.window_size),
                    axis=1)

            else:
                padded_sequence = to_categorical(padded_sequence, num_classes=len(tokenizer.index_word) + 1)
            i = i + 1

            p = model.predict(
                x=np.array(padded_sequence).reshape((1, args.window_size, len(tokenizer.index_word) + 1 + (1 if year else 0) + 1))
            )
            # probs = list(p[0])
            # print(p)
            last_char = np.random.choice(words, p=p.reshape((96,)))
            # last_char = words[probs.index(max(probs))]

            sequence += [last_char]
            out_sequence += [last_char]

        print('{} -> {}'.format(
            ''.join(initial_sequence),
            '{}{}'.format(''.join(initial_sequence), ''.join(out_sequence[:-1])))
        )
        out.append('{}{}\n'.format(''.join(initial_sequence), ''.join(out_sequence[:-1])))


with open('data/output/2000-bNONVEG-{}.txt'.format(args.configuration), 'w+', encoding='utf-8') as f:
    f.writelines(out)
