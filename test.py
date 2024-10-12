import keras
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras import layers
from tensorflow.python.keras.backend import dtype


def main():
    print("Start hello world ML program...")

    l0 = layers.Dense(units = 1)

    model = Sequential()
    model.add(keras.Input(shape=(1,)))
    model.add(l0)
    model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

    xs = np.array([-1, 0, 1, 2, 3, 4], dtype = float)
    ys = np.array([-3, -1, 1, 3, 5, 7], dtype = float)

    # Almost by 400 tries the error comes close to zero.
    model.fit(xs, ys, epochs=500)

    x = 10
    print("\nAnswer for x= {} is {}".format(x, model.predict(np.array([x], dtype = float))))

    print("\nModel L0 learning -> {}".format(l0.get_weights()))

if __name__ == '__main__':
    main()