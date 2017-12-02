import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
from skimage import io
from PIL import Image
import pandas as pd
from tqdm import tqdm
import time
import os

from models import *
from logger import Logger

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])
transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])


class Cifar10Loader(Dataset):
    def __init__(self, transform=None, train=True):
        self.train = train
        self.transform = transform
        self.label_frame = pd.read_csv('/data/datasets/yanfu/kaggle_cifar10/trainLabels.csv')
        self.train_dir = '/data/datasets/yanfu/kaggle_cifar10/train'
        self.train_list = os.listdir(self.train_dir)
        self.valid_dir = '/data/datasets/yanfu/kaggle_cifar10/valid'
        self.valid_list = os.listdir(self.valid_dir)
        self.classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
                        'ship': 8, 'truck': 9}

    def __getitem__(self, index):
        if self.train:
            img_name = os.path.join(self.train_dir, self.train_list[index])
        else:
            img_name = os.path.join(self.valid_dir, self.valid_list[index])

        # img = io.imread(img_name)
        img = Image.open(img_name).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # get label from the csv file
        img_id = int(img_name.split('/')[-1].split('.')[0])
        label_class = self.label_frame.ix[img_id - 1].label
        label = self.classes[label_class]
        return img, label

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.valid_list)


# params to tune
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
end_epoch = 350
batch_size = 1024
net_name = 'VGG19'

train_set = Cifar10Loader(transform=transform_train, train=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
valid_set = Cifar10Loader(transform=transform_train, train=False)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4)

net = VGG(net_name)
# net = ResNet101()
net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)  # not good, validation acc 0.7 @ 100 epochs

criterion = nn.CrossEntropyLoss()
best_acc = 0  # best test accuracy
step = 0
logger = Logger('./logs')


def train(epoch):
    global step
    start = time.clock()
    print('*** epoch: {} ***'.format(epoch))
    net.train()
    train_loss = 0
    correct_num = 0
    pred_num = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, pred = torch.max(outputs.data, 1)
        pred_num += targets.size(0)
        correct_num += pred.eq(targets.data).cpu().sum()

        # if (batch_idx + 1) % 100:
        print('train: loss {:.3f}, acc {:.3f}, lr {}'.format(train_loss / (batch_idx + 1), correct_num / pred_num,
                                                             optimizer.param_groups[0]['lr']))

        info = {'train_loss': train_loss / (batch_idx + 1),
                'train_acc': correct_num / pred_num,
                'lr': optimizer.param_groups[0]['lr']}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)
        step += 1
    print('*** epoch ends, took {}s ***'.format(time.clock() - start))


def valid(epoch):
    global best_acc
    net.eval()
    valid_loss = 0
    pred_num = 0
    correct_num = 0
    for batch_idx, (inputs, labels) in enumerate(valid_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.data[0]
        _, pred = torch.max(outputs.data, 1)
        pred_num += labels.size(0)
        correct_num += pred.eq(labels.data).cpu().sum()
        if (batch_idx + 1) % 10:
            print('valid: loss {:.3f}, acc {:.3f}'.format(valid_loss / (batch_idx + 1), correct_num / pred_num))

    acc = correct_num / pred_num
    info = {'valid_loss': valid_loss / (batch_idx + 1), 'valid_acc': acc}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step + 1)
    if acc > best_acc:
        print('got best acc, saving...')
        state = {
            'net': net.module,
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + net_name + '_epoch' + str(epoch) + '.t7')
        best_acc = acc


for epoch in range(start_epoch, end_epoch):
    scheduler.step()
    train(epoch)
    valid(epoch)
