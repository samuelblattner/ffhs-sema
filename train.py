from build import build_model
from loader import Loader

BATCH_SIZE = 1000
NUM_BATCHES = 5
NUM_EPOCHS = 10

model, tokenizer, max_num_chars = build_model()

loader = Loader(
    batch_size=BATCH_SIZE,
    offset=0,
    limit=NUM_BATCHES * BATCH_SIZE
)

validation_loader = Loader(
    batch_size=BATCH_SIZE,
    offset=BATCH_SIZE * NUM_BATCHES,
    limit=NUM_BATCHES * BATCH_SIZE
)

gen = loader.get_train_generator(tokenizer, max_num_chars)

model.fit_generator(
    generator=loader.get_train_generator(tokenizer, max_num_chars),
    steps_per_epoch=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=validation_loader.get_train_generator(tokenizer, max_num_chars),
    validation_steps=1
)
