

# gooyoo




import tensorflow as tf;
from tensorflow import keras;
from tensorflow.keras import layers;
from tensorflow.keras.datasets import mnist;




'''
physical_devices = tf.config.list_physical_devices("GPU");
tf.config.experimental.set_memory_growth(physical_devices[0], True);
'''

(x_train, y_train), (x_test, y_test) = mnist.load_data();
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0;
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0;




model = keras.Sequential(
    [
     layers.Dense(64, activation="relu"), layers.Dense(10)
     ]
    );




inputs = keras.Input(784);
x = layers.Dense(64, activation = "relu")(inputs);
outputs = layers.Dense(10)(x);
model = keras.Model(inputs = inputs, outputs = outputs);




model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(),
    metrics = ["accuracy"],
);

model.fit(x_train, y_train, batch_size = 32, epochs = 2, verbose = 1);
model.evaluate(x_test, y_test, batch_size = 32, verbose = 2);
# model.save_weights('checkpoint_folder/')
#model.save("saved_model/");
