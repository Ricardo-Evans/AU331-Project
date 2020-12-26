# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Dtiv7WA2lQSYfpVBVINNrIO5WAX1ZM54
"""

# from google.colab import drive
# drive.mount('/content/drive')

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision
import torch
from torchvision import datasets, transforms

data_train = datasets.MNIST(root = "./Mnist/", transform=transforms,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root="./Mnist/",
                           transform = transforms,
                           train = False, download = True)

class Mnist_Loader(object):
  class_num = 10
  DATA_SIZE = (60000,10000)
  FIG_W = 45
  def __init__(self, flatten=False, data_path = 'mnist'):
    self.flatten = flatten
    self.load_data(data_path)
    self.data_train = (self.data_train - self.data_train.mean())/ self.data_train.std()
    self.data_test = (self.data_test - self.data_test.mean()) / self.data_test.std()
  
  def load_data(self, data_path = 'mnist'):
    self.data_train = np.concatenate([np.load(os.path.join(data_path,'mnist_train_data_part1.npy')), np.load(os.path.join(data_path, 'mnist_train_data_part2.npy'))], axis=0).astype(np.float32)
    self.label_train = np.load(os.path.join(data_path, 'mnist_train_label.npy')).astype(np.int64)
    self.data_test = np.load(os.path.join(data_path, 'mnist_test_data.npy')).astype(np.float32)
    self.label_test = np.load(os.path.join(data_path, 'mnist_test_label.npy')).astype(np.int64)
    if self.flatten:
      self.data_train = self.data_train.reshape(self.data_train.shape[0], -1)
      self.data_test = self.data_test.reshape(self.data_test.shape[0], -1)
  def demo(self):
    print('Train data:', self.data_train.shape)
    print('Train labels:', self.label_train.shape)
    print('Test data:', self.data_test.shape)
    print('Test labels:', self.label_test.shape)

    ind = np.random.randint(0, self.DATA_SIZE[0])

    # print the index and label
    print("index: ", ind)
    print("label: ", self.label_train[ind])
    print("data: ", self.data_train[ind])
    # save the figure
    plt.imshow(self.data_train[ind].reshape(self.FIG_W, self.FIG_W))
    plt.show()

data = Mnist_Loader(flatten = False, data_path='/content/drive/MyDrive/Colab Notebooks/mnist')

data.demo()



from PIL import Image
from torch.utils.data.dataset import Dataset
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
# class Bottleneck(nn.Module):
#     def __init__(self, in_planes, growth_rate):
#         super(Bottleneck, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(4*growth_rate)
#         self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = torch.cat([out,x], 1)
#         return out


# class Transition(nn.Module):
#     def __init__(self, in_planes, out_planes):
#         super(Transition, self).__init__()
#         self.bn = nn.BatchNorm2d(in_planes)
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

#     def forward(self, x):
#         out = self.conv(F.relu(self.bn(x)))
#         out = F.avg_pool2d(out, 2)
#         return out


# class DenseNet(nn.Module):
#     def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
#         super(DenseNet, self).__init__()
#         self.growth_rate = growth_rate

#         num_planes = 2*growth_rate
#         self.conv1 = nn.Conv2d(1, num_planes, kernel_size=3, padding=1, bias=False)

#         self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
#         num_planes += nblocks[0]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans1 = Transition(num_planes, out_planes)
#         num_planes = out_planes

#         self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
#         num_planes += nblocks[1]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans2 = Transition(num_planes, out_planes)
#         num_planes = out_planes

#         self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
#         num_planes += nblocks[2]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans3 = Transition(num_planes, out_planes)
#         num_planes = out_planes

#         self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
#         num_planes += nblocks[3]*growth_rate

#         self.bn = nn.BatchNorm2d(num_planes)
#         self.linear = nn.Linear(num_planes, num_classes)

#     def _make_dense_layers(self, block, in_planes, nblock):
#         layers = []
#         for i in range(nblock):
#             layers.append(block(in_planes, self.growth_rate))
#             in_planes += self.growth_rate
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.trans1(self.dense1(out))
#         out = self.trans2(self.dense2(out))
#         out = self.trans3(self.dense3(out))
#         out = self.dense4(out)
#         out = F.avg_pool2d(F.relu(self.bn(out)), 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

# def DenseNet121():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

# def DenseNet169():
#     return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

# def DenseNet201():
#     return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

# def DenseNet161():
#     return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, 512)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

class SiameseSVMNet_dense(nn.Module):
  def __init__(self):
    super(SiameseSVMNet_dense, self).__init__()
    self.featureNet = DenseNet121()
    self.fc = nn.Linear(32, 1)

  def forward(self, x1, x2):
    output1 = self.featureNet(x1)
    output2 = self.featureNet(x2)
    difference = torch.abs(output1 - output2)
    print (difference.shape)
    output = self.fc(difference)
    return output

class SVMLoss(nn.Module):
  def __init__(self, C):
    super(SVMLoss, self).__init__()
    self.C = C

  def forward(self, output, y):                #利用svm最小化的函数表示Loss
    temp1 = (output * y).view(-1)
    ones = temp1 / temp1
    temp2 = ones - temp1
    zeros = temp2 - temp2
    elementwiseloss = torch.max(temp2, zeros)
    elementwiseloss = elementwiseloss * elementwiseloss
    return self.C * torch.mean(elementwiseloss)


def compute_accuracy(predictions, labels):
    predictions = torch.sign(predictions)
    correct = predictions.eq(labels)
    result = correct.sum().data.cpu()
    return result

def get_loaders(args):
  data_train = MakePair(data.data_train, data.label_train)
  if args.cuda:
    train_loader = DataLoader(data_train,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=1,
                                pin_memory=True)
  else:
    train_loader = DataLoader(data_train,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=False)
  data_test = data.data_test
  return train_loader, data_test

class MakePair(object):
  def __init__(self, data_train, label_train, class_num=10, transform=None):
    self.data = []
    for i in range(class_num):
      self.data.append([])
    for i in range(60000):
      self.data[label_train[i]].append(data_train[i])

  def __getitem__(self, index):
    class_choose = index % 10
    length = index % 5000
    if (index % 2 == 0):
      data1 = self.data[class_choose][length]
      if length == 0:
        num = random.randrange(1, len(self.data[class_choose]))
      else:
        num = random.randrange(0, length)
      data2 = self.data[class_choose][num]
      data1 = np.expand_dims(data1, 0)
      data2 = np.expand_dims(data2,0)
      return data1, data2, torch.from_numpy(np.array([1])).float()
    else:
      data1 =self.data[class_choose][length]
      inc = random.randrange(1, len(self.data))
      rand1 = (class_choose + inc) % len(self.data)  #随机选取不同类
      num = random.randrange(0, len(self.data[rand1]))
      data2 = self.data[rand1][num]
      data1 = np.expand_dims(data1,0)
      data2 = np.expand_dims(data2,0)
      return data1, data2, torch.from_numpy(np.array([-1])).float()

  def __len__(self):
    num = 0
    for i in range(10):
      num += len(self.data[i])
    return num



import argparse
from torchvision import transforms
from sklearn.metrics import accuracy_score
# constants
dim = 105

class FeatureNet(nn.Module):       #提取feature
  def __init__(self):
    super(FeatureNet, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=2, padding=0),
        nn.ReLU(inplace=True),
        nn.AdaptiveMaxPool2d((12,12))
    )
        
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=16, kernel_size=2, stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.AdaptiveMaxPool2d((10,10))
    )
    
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )
        
    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveMaxPool2d((5,5))
    )

    self.fc1 = nn.Sequential(
        nn.Linear(1600,800),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(x.size(0),-1)
    x = self.fc1(x)
    return x

def get_args():
  parser = argparse.ArgumentParser(description='Siamese network SVM on Mnist')
  parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                      help='input batch size for training (default: 256)')
  parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                      help='input batch size for testing (default: 256)')
  parser.add_argument('--epochs', type=int, default=100, metavar='N',
                      help='number of epochs to train (default: 20)')
  parser.add_argument('--C', type=float, default=0.2, metavar='C',
                      help='C regulation coefficients (default: 0.2)')
  parser.add_argument('--test-number', type=int, default=10, metavar='N',
                      help='number of different subset of test (default: 10)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA training')
  parser.add_argument('-f')
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  return args

class MLKD_Net(nn.Module):
  def __init__(self):
    super(MLKD_Net, self).__init__()
    self.featureNet = FeatureNet() #FeatureNet()
    self.fc = nn.Linear(800 ,1)

  def forward(self, x1, x2):
    output1 = self.featureNet(x1)
    output2 = self.featureNet(x2)
    difference = torch.abs(output1 - output2)
    output = self.fc(difference)
    return output

  def get_FeatureNet(self):
    return self.featureNet

args = get_args()
model = MLKD_Net()
if args.cuda:
  model = model.cuda()
criterion = SVMLoss(args.C)
optimizer = torch.optim.Adam(params=model.parameters())
train_loader, test_loader = get_loaders(args)
def training(epoch, loss_epoch):
  print('Epoch', epoch + 1)
  model.train()
  # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
  acc_num = 0
  count =0 
  for batch_idx, (x0, x1, label) in enumerate(train_loader):
    if args.cuda:
      x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
    x0, x1, label = Variable(x0), Variable(x1), Variable(label)
    output = model(x0, x1)
    out = []
    for si in output:
      if si <= 0:
        out.append(-1)
      else:
        out.append(1)
    out = Variable(torch.from_numpy(np.array(out)).cuda())
    acc_num += output.shape[0] * accuracy_score(out.cpu().detach().numpy(), label.cpu().detach().numpy())
    count += output.shape[0]
    optimizer.zero_grad()
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    model.eval()
  accuracy = acc_num / count
  print ("epoch %d accuracy is", epoch, accuracy)
  return accuracy
best_val = 0.0
test_results = []
train_result = []
epoch_list = []
loss_list = []
for epoch in range(args.epochs):
  train_accuracy = training(epoch, loss_list)
  epoch_list.append(epoch)
  train_result.append(train_accuracy)
# Print best results
# print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
#     test_results[0]))
# print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
#     test_results[1]))
# print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
#     test_results[2]))
# print('\nResult: 5 way 1 shot Accuracy: {:.4f}%\n'.format(
#     test_results[3]))

plt.plot(train_result)

test_results = []
train_results = []
from sklearn import svm
from sklearn.metrics import accuracy_score
def test(data, i):
  model.eval()
  clf = svm.SVC(C=args.C, kernel='linear', decision_function_shape='ovo')
  featuremodel = model.get_FeatureNet()
  if args.cuda:
    featuremodel = featuremodel.cuda()
  # choose classes
  acc = 0
  train_features = []
  test_features = []
  for data_train in data.data_train[6000 * i: 6000* (i+1)]:
    data_train = data_train.reshape(1,1,data_train.shape[0],data_train.shape[1])
    data_train = torch.from_numpy(data_train).float()
    if args.cuda:
      data_train = data_train.cuda()
    train_features.append(featuremodel(Variable(data_train)).cpu().data.numpy())
  for data_test in data.data_test:
    data_test = data_test.reshape(1,1,data_test.shape[0],data_test.shape[1])
    data_test = torch.from_numpy(data_test).float()
    if args.cuda:
      data_test = data_test.cuda()
    test_features.append(featuremodel(Variable(data_test)).cpu().data.numpy())
  train_loader, test_loader = get_loaders(args)
  count = 0
  acc_num = 0
  for batch_idx, (x0, x1, label) in enumerate(train_loader):
    if args.cuda:
      x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
    x0, x1, label = Variable(x0), Variable(x1), Variable(label)
    output = model(x0, x1)
    # print(output.shape)
    # print(label.shape)
    out = []
    for si in output:
      if si < 0:
        out.append(-1)
      else:
        out.append(1)
    out = Variable(torch.from_numpy(np.array(out)).cuda())
    acc_num += output.shape[0] * accuracy_score(out.cpu().detach().numpy(), label.cpu().detach().numpy())
    count += output.shape[0]
  accuracy_train = acc_num/count
    # create features
  train_features = np.array(train_features)
  train_features = np.reshape(
      train_features, (train_features.shape[0], 800))
  test_features = np.array(test_features)
  test_features = np.reshape(
      test_features, (test_features.shape[0], 800))
  # predict with SVM
  print ("enter svm")
  clf.fit(train_features, data.label_train[6000*i :6000*(i+1) ])
  print ("predict")
  pred = clf.predict(test_features)

  acc = accuracy_score(data.label_test, pred)
  
  acc = 100.0*acc
  
  return acc, accuracy_train
for epoch in range(10):
  print ("this is epoch " , epoch)
  test_accuracy, train_accuracy = test(data, epoch)
  test_results.append(test_accuracy)
  train_results.append(train_accuracy)



plt.plot((1,2,3,4,5,6,7,8,9,10), test_results)

class MLKD_ResNet(nn.Module):
  def __init__(self):
    super(MLKD_ResNet, self).__init__()
    self.featureNet = ResNet18() #FeatureNet()
    self.fc = nn.Linear(512 ,1)

  def forward(self, x1, x2):
    output1 = self.featureNet(x1)
    output2 = self.featureNet(x2)
    difference = torch.abs(output1 - output2)
    output = self.fc(difference)
    return output

  def get_FeatureNet(self):
    return self.featureNet

!pip install GPUtil

!pip install psutil

import GPUtil
import time
import psutil
Gpus = GPUtil.getGPUs()
for gpu in Gpus:
  print('gpu.id:', gpu.id)
  print('GPU总量：', gpu.memoryTotal)
  print('GPU使用量：', gpu.memoryUsed)
  print('gpu使用占比:', gpu.memoryUtil * 100)

torch.cuda.empty_cache()
for gpu in Gpus:
  print('gpu.id:', gpu.id)
  print('GPU总量：', gpu.memoryTotal)
  print('GPU使用量：', gpu.memoryUsed)
  print('gpu使用占比:', gpu.memoryUtil * 100)

