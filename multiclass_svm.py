import kernel
import numpy as np
import keras
import pandas as pd
from sklearn import svm
import random

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


class multiclass_svm:
    def __init__(self, dataset, classes):
        # datasets should be shuffled
        self.classes = classes
        self.datasets = dataset
        self.sub_svm_list = [[] for _ in range(self.classes - 1)]
        # sub_svm_list needs classes-1 entries
        # sub_svm_list[i] needs classes-i-1 entries

    def split_datasets(self, ratio):
        size = len(self.datasets)
        train_dataset = self.datasets[:int(ratio * size)]
        self.test_dataset = self.datasets[int(ratio * size) + 1:]
        self.cat_train_dataset = [[] for _ in range(self.classes)]
        for example in train_dataset:
            self.cat_train_dataset[example[-1]].append(example)

    def train(self):
        # train one svm for each label pair
        for label1 in range(self.classes):
            for label2 in range(label1 + 1, self.classes):
                sub_data1 = self.cat_train_dataset[label1]
                sub_data2 = self.cat_train_dataset[label2]
                sub_train_data1 = np.array(sub_data1).astype("float64")
                sub_train_data2 = np.array(sub_data2).astype("float64")
                sub_train_data = np.concatenate([sub_train_data1, sub_train_data2], axis=0)
                np.random.shuffle(sub_train_data)
                sub_train_feature = sub_train_data[:, :-1]
                sub_train_label = sub_train_data[:, -1]

                kernel_function = kernel.kernel(sub_train_feature, sub_train_label, sub_train_feature.shape[1])
                sub_clf = svm.SVC(kernel=kernel_function)
                #sub_clf = svm.SVC(kernel="rbf")

                sub_clf.fit(sub_train_feature, sub_train_label)
                self.sub_svm_list[label1].append(sub_clf)
        print("Training finished")

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


class multiclass_svm_modified:
    def __init__(self, dataset, classes):
        # datasets should be shuffled
        self.classes = classes
        self.datasets = dataset
        self.sub_svm_dict = {}
        # one for every split

    def split_datasets(self, ratio):
        size = len(self.datasets)
        train_dataset = self.datasets[:int(ratio * size)]
        self.test_dataset = self.datasets[int(ratio * size) + 1:]
        self.cat_train_dataset = [[] for _ in range(self.classes)]
        for example in train_dataset:
            self.cat_train_dataset[example[-1]].append(example)

    def train(self):
        self._train(0, self.classes - 1)

    def _train(self, s, e):
        if s >= e:
            return
        sp = (s + e) // 2
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

        kernel_function = kernel.kernel(sub_train_feature, sub_train_label, sub_train_feature.shape[1])
        sub_clf = svm.SVC(kernel=kernel_function)

        # sub_clf = svm.SVC(kernel="rbf")
        sub_clf.fit(sub_train_feature, sub_train_label)
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


if __name__ == "__main__":

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

    my_multi_svm = multiclass_svm_modified(datas, 3)
    my_multi_svm.split_datasets(0.8)
    my_multi_svm.train()
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
