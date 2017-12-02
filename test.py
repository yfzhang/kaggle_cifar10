import torch
import torchvision
import torchvision.transforms as transforms

from models import *
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
from skimage import io
import pandas as pd
from tqdm import tqdm
import time
import os

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class Cifar10Loader(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.test_dir = '/data/datasets/yanfu/kaggle_cifar10/test'
        self.test_list = os.listdir(self.test_dir)

    def __getitem__(self, index):
        img_name = os.path.join(self.test_dir, self.test_list[index])
        img_idx = int(img_name.split('/')[-1].split('.')[0])
        img = io.imread(img_name)
        if self.transform:
            img = self.transform(img)
        return img, img_idx

    def __len__(self):
        return len(self.test_list)


batch_size = 512
checkpoint_file = './checkpoint/ResNet101_epoch276.t7'

test_set = Cifar10Loader(transform=transform_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

print('loading weights...')
assert os.path.isdir('checkpoint'), 'no checkpoint folder found'
checkpoint = torch.load(checkpoint_file)
net = checkpoint['net']
net.cuda()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True
classes = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
           9: 'truck'}
# test
net.eval()
preds = []
img_ids = []
for batch_idx, (inputs, img_idx) in enumerate(tqdm(test_loader)):
    inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)
    _, pred = torch.max(outputs.data, 1)
    preds.extend(torch.max(outputs.data, 1)[1])
    img_ids.extend(img_idx)

df = pd.DataFrame({'id': img_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: classes[x])
# df = df.sort_values(by=['id'])
df.to_csv('test.csv', index=False)
