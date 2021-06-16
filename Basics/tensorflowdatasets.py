

# gooyoo




# !pip install tensorflow_datasets
import tensorflow as tf;
from tensorflow import keras;
from tensorflow.keras import layers;
import tensorflow_datasets as tfds;




'''
physical_devices = tf.config.list_physical_devices("GPU");
tf.config.experimental.set_memory_growth(physical_devices[0], True);
'''

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split = ["train", "test"],
    shuffle_files = True,
    as_supervised = True,  
    with_info = True,  
    );


fig = tfds.show_examples(ds_train, ds_info, rows = 4, cols = 4);
print(ds_info);


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label;




AUTOTUNE = tf.data.experimental.AUTOTUNE;
BATCH_SIZE = 128;


# Setup for train dataset

ds_train = ds_train.map(normalize_img, num_parallel_calls = AUTOTUNE);
ds_train = ds_train.cache();
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples);
ds_train = ds_train.batch(BATCH_SIZE);
ds_train = ds_train.prefetch(AUTOTUNE);


# Setup for test Dataset

ds_test = ds_train.map(normalize_img, num_parallel_calls = AUTOTUNE);
ds_test = ds_train.batch(128);
ds_test = ds_train.prefetch(AUTOTUNE);




model = keras.Sequential(
    [
        keras.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation = "relu"),
        layers.Flatten(),
        tf.keras.layers.Dense(10, activation = "softmax"),
    ]
);


model.compile(
    optimizer = keras.optimizers.Adam(0.001),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"],
);

model.fit(ds_train, epochs = 5, verbose = 1);
model.evaluate(ds_test);





