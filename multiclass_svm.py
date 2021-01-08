import kernel
import numpy as np
import keras
import pandas as pd
from sklearn import svm
import random
from torchvision import datasets, transforms
import torch
import tqdm
import cv2
import time

'''
def multiclass_svm(datasets,classes):
    # one vs rest method
    """
    :param datasets:
        dict, "data":(size,dim) float64, "target":(size,) float64 [0,..,size-1]

    :return: list of svm, number=size
        i-th svm is for label i and the rest
    """
'''


def transfer(x):
    # x: 28*28
    x = x.numpy().astype("uint8")
    x = cv2.resize(x, (5, 5))
    _, thresh = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
    return (torch.tensor(thresh) // 255).view(1, 25)


def get_mnist_data():
    trainset = datasets.MNIST(root="./datasets/mnist", transform=transforms.ToTensor(), train=True,
                              download=True)
    testset = datasets.MNIST(root="./datasets/mnist", transform=transforms.ToTensor(), train=False,
                             download=True)

    # x = torch.cat([trainset.train_data.view(60000, -1), testset.test_data.view(10000, -1)], 0)
    y = torch.cat([trainset.train_labels, testset.test_labels], 0).view(-1, 1).int()

    x = torch.cat([
                      transfer(trainset.train_data[i]) for i in range(60000)
                  ] + [transfer(testset.test_data[i]) for i in range(10000)
                       ], dim=0)

    datas = torch.cat([x, y], 1).numpy().tolist()
    random.shuffle(datas)
    return datas


def get_iris_data():
    filename = "datasets/iris/iris.data"
    irismap = {"Iris-setosa": 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    datas = []
    with open(filename) as f:
        raw_data = f.readlines()
    data_tmp = [line.strip().split(",") for line in raw_data]
    for exmaple in data_tmp:
        feature = [float(i) for i in exmaple[:4]]
        label = irismap[exmaple[4]]
        datas.append(feature + [label])
    random.shuffle(datas)
    return datas


def get_car_data():
    filename = "datasets/car/car.data"
    carmap = {"unacc": 0, 'acc': 1, 'good': 2, "vgood": 3}
    buyingmap = {"vhigh": 9, "high": 1, "med": 2, "low": 3}
    maintmap = {"vhigh": 0, "high": 1, "med": 2, "low": 3}
    doorsmap = {"2": 0, "3": 1, "4": 2, "5more": 3}
    personsmap = {"2": 0, "4": 1, "more": 2}
    lug_bootmap = {"small": 0, "med": 1, "big": 2}
    safetymap = {"low": 0, "med": 1, "high": 2}
    feature_maps = [buyingmap, maintmap, doorsmap, personsmap, lug_bootmap, safetymap]

    datas = []
    with open(filename) as f:
        raw_data = f.readlines()
    data_tmp = [line.strip().split(",") for line in raw_data]
    for exmaple in data_tmp:
        feature = [feature_maps[i][exmaple[i]] for i in range(6)]
        label = carmap[exmaple[6]]
        datas.append(feature + [label])
    random.shuffle(datas)
    return datas


def get_abalone_data():
    filename = "datasets/abalone/abalone.data"
    sexmap = {"M": 0, "F": 1, "I": 2}
    datas = []
    with open(filename) as f:
        raw_data = f.readlines()
    data_tmp = [line.strip().split(",") for line in raw_data]
    for exmaple in data_tmp:
        feature = [sexmap[exmaple[0]]]
        feature += [float(i) for i in exmaple[1:8]]
        label = int(exmaple[8])
        if label == 29:
            label -= 2
        else:
            label -= 1
        datas.append(feature + [label])
    random.shuffle(datas)
    return datas


class multiclass_svm:
    def __init__(self, dataset, classes, nn_kernel):
        print("Multiclass svm initialization -- START")
        # datasets should be shuffled
        self.classes = classes
        self.datasets = dataset
        self.nn_kernel = nn_kernel
        self.sub_svm_list = [[] for _ in range(self.classes - 1)]
        # sub_svm_list needs classes-1 entries
        # sub_svm_list[i] needs classes-i-1 entries
        print("Multiclass svm initializing -- END")

    def split_datasets(self, ratio):
        print("Multiclass svm dataset split -- START")
        size = len(self.datasets)
        train_dataset = self.datasets[:int(ratio * size)]
        self.test_dataset = self.datasets[int(ratio * size) + 1:]
        self.cat_train_dataset = [[] for _ in range(self.classes)]
        for example in train_dataset:
            self.cat_train_dataset[example[-1]].append(example)
        print("Multiclass svm dataset split -- END")

    def train(self):
        print("Multiclass svm training -- START")
        # train one svm for each label pair
        for label1 in range(self.classes):
            for label2 in range(label1 + 1, self.classes):
                print("Training step : ", label1, " vs ", label2)
                sub_data1 = self.cat_train_dataset[label1]
                sub_data2 = self.cat_train_dataset[label2]
                sub_train_data1 = np.array(sub_data1).astype("float64")
                sub_train_data2 = np.array(sub_data2).astype("float64")
                sub_train_data = np.concatenate([sub_train_data1, sub_train_data2], axis=0)
                np.random.shuffle(sub_train_data)
                sub_train_feature = sub_train_data[:, :-1]
                sub_train_label = sub_train_data[:, -1]

                if self.nn_kernel:
                    print("Training NN kernel --- START")
                    kernel_function = kernel.kernel(sub_train_feature, sub_train_label, sub_train_feature.shape[1])
                    sub_clf = svm.SVC(kernel=kernel_function)
                    print("Training NN kernel --- END")
                else:
                    sub_clf = svm.SVC(kernel="rbf")
                print("Training sub SVM --- START")
                sub_clf.fit(sub_train_feature, sub_train_label)
                print("Training sub SVM --- END")
                self.sub_svm_list[label1].append(sub_clf)
        print("Multiclass svm training -- END")

    def _get_predict(self, feature):
        vote_res = np.zeros([self.classes])
        for label1 in range(self.classes - 1):
            for label2 in range(label1 + 1, self.classes):
                sub_vote = self.sub_svm_list[label1][label2 - label1 - 1].predict(feature)
                sub_vote_index = int(sub_vote[0])
                vote_res[sub_vote_index] += 1
        label_ = np.argmax(vote_res)
        return label_

    def eval(self):
        print("Multiclass svm evaluation -- START")
        total = len(self.test_dataset)
        right = 0.0
        for example in self.test_dataset:
            feature = np.array([example[:-1]]).astype("float64")
            label = example[-1]
            label_ = self._get_predict(feature)
            if label_ == label:
                right += 1
        acc = right / total
        print("Accuracy: ", acc)
        print("Multiclass svm evaluation -- END")


class multiclass_svm_modified:
    def __init__(self, dataset, classes, nn_kernel):
        print("Multiclass svm initialization -- START")
        # datasets should be shuffled
        self.classes = classes
        self.datasets = dataset
        self.nn_kernel = nn_kernel
        self.sub_svm_dict = {}
        print("Multiclass svm initializing -- END")
        # one for every split

    def split_datasets(self, ratio):
        print("Multiclass svm dataset split -- START")
        size = len(self.datasets)
        train_dataset = self.datasets[:int(ratio * size)]
        self.test_dataset = self.datasets[int(ratio * size) + 1:]
        self.cat_train_dataset = [[] for _ in range(self.classes)]
        for example in train_dataset:
            self.cat_train_dataset[example[-1]].append(example)
        print("Multiclass svm dataset split -- END")

    def train(self):
        print("Multiclass svm training -- START")
        self._train(0, self.classes - 1)
        print("Multiclass svm training -- END")

    def _train(self, s, e):
        if s >= e:
            return
        sp = (s + e) // 2
        print("Training step : ", s,"-",sp, " vs ", sp+1,"-",e)
        first_half = []
        second_half = []
        for part in self.cat_train_dataset[s: sp + 1]:
            first_half += part
        for part in self.cat_train_dataset[sp + 1: e + 1]:
            second_half += part
        for example in first_half:
            example[-1] = 0.0
        for example in second_half:
            example[-1] = 1.0
        sub_train_data = np.concatenate([np.array(first_half), np.array(second_half)], axis=0).astype("float64")
        np.random.shuffle(sub_train_data)
        sub_train_feature = sub_train_data[:, :-1]
        sub_train_label = sub_train_data[:, -1]

        if self.nn_kernel:
            print("Training NN kernel --- START")
            kernel_function = kernel.kernel(sub_train_feature, sub_train_label, sub_train_feature.shape[1])
            sub_clf = svm.SVC(kernel=kernel_function)
            print("Training NN kernel --- END")
        else:
            sub_clf = svm.SVC(kernel="rbf")
        print("Training sub SVM --- START")
        sub_clf.fit(sub_train_feature, sub_train_label)
        print("Training sub SVM --- START")
        self.sub_svm_dict[(s, sp, e)] = sub_clf

        self._train(s, sp)
        self._train(sp + 1, e)

    def get_predict(self, feature):
        return self._get_predict(0, self.classes - 1, feature)

    def _get_predict(self, s, e, feature):
        if s >= e:
            assert s == e
            return s
        else:  # need to classify
            sp = (s + e) // 2
            sub_svm = self.sub_svm_dict[(s, sp, e)]
            res = sub_svm.predict(feature)
            if res == 0:
                return self._get_predict(s, sp, feature)
            else:
                return self._get_predict(sp + 1, e, feature)

    def eval(self):
        print("Multiclass svm evaluation -- START")
        total = len(self.test_dataset)
        right = 0.0
        for example in self.test_dataset:
            feature = np.array([example[:-1]]).astype("float64")
            label = example[-1]
            label_ = self.get_predict(feature)
            if label_ == label:
                right += 1
        acc = right / total
        print("Accuracy: ", acc)
        print("Multiclass svm evaluation -- END")


if __name__ == "__main__":
    datas_iris = get_iris_data() # 3
    datas_abalone = get_abalone_data() # 28
    datas_car = get_car_data() # 4


    my_multi_svm = multiclass_svm(datas_abalone, 28, nn_kernel=True)
    my_multi_svm.split_datasets(0.8)
    start=time.time()
    my_multi_svm.train()
    end=time.time()
    print("Training time: ",end-start," seconds")
    my_multi_svm.eval()

    '''
    tr=np.array(datas[:120]).astype("float64")
    te=np.array(datas[121:]).astype("float64")
    trx=tr[:,:-1]
    tr_y=tr[:,-1]
    tex=te[:,:-1]
    tey=te[:,-1]
    clf = svm.SVC()
    clf.fit(trx,tr_y)
    print("accuracy:", clf.score(tex, tey))
    '''
