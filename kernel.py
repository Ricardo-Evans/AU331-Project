import keras
import numpy as np
import functools
import math


def process(x1, x2):
    length = len(x1) + len(x2)
    if length != len(x1) * 2 or length != len(x2) * 2:
        raise
    data = np.zeros((length,), np.float)
    for i in range(0, int(length / 2)):
        data[i] = x1[i] * x2[i]
    for i in range(int(length / 2), length):
        data[i] = abs(x1[i - int(length / 2)] - x2[i - int(length / 2)])
    return data


def combine(x, y):
    length = len(y)
    result_x = []
    result_y = []
    for i in range(0, length):
        for j in range(i, length):
            result_x.append(process(x[i], x[j]))
            result_y.append(1 if y[i] == y[j] else 0)
    return np.asarray(result_x), np.asarray(result_y)


def kernel_matrix(x1, x2, model):
    result = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            result[i][j] = model.predict(np.asarray([process(x1[i], x2[j])]))[0][0]
    return result


def kernel(x, y, vector_size):
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(vector_size * 2,)),
        keras.layers.Dense(vector_size * 2, activation="sigmoid", kernel_initializer='he_uniform'),
        keras.layers.Dense(vector_size * 2, activation="sigmoid", kernel_initializer='he_uniform'),
        keras.layers.Dense(vector_size * 2, activation="sigmoid", kernel_initializer='he_uniform'),
        keras.layers.Dense(vector_size * 2, activation="sigmoid", kernel_initializer='he_uniform'),
        keras.layers.Dense(1, activation="sigmoid", kernel_initializer="he_uniform")
    ])
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam")
    x_, y_ = combine(x, y)
    model.fit(x_, y_)
    return functools.partial(kernel_matrix, model=model)
