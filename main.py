import kernel
import numpy as np
import keras
import pandas as pd
from sklearn import svm
import time


def main():
    dataset = pd.read_csv("./datasets/data.csv")
    dataset.drop(columns='id', inplace=True)
    dataset.dropna(axis=1, how='all', inplace=True)
    dataset.replace('M', 1, inplace=True)
    dataset.replace('B', 0, inplace=True)
    data = dataset.values
    np.random.shuffle(data)
    y, x = np.split(data, (1,), axis=1)
    y = y.flatten()
    ratio = 0.8
    index = int(ratio * len(data))
    x_train, x_test, y_train, y_test = x[:index], x[index:], y[:index], y[index:]
    kernel_function = kernel.kernel(x_train, y_train, 30)
    clf = svm.SVC(kernel=kernel_function)

    a=time.time()
    clf.fit(x_train, y_train.ravel())
    b=time.time()
    print("time: ",b-a," seconds")

    print("accuracy:", clf.score(x_test, y_test))


if __name__ == '__main__':
    main()
