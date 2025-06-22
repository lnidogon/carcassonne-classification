from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import numpy as np
import torchvision
import torch
import os
import cv2
import numpy as np
import csv

class CarcMetricDataset(Dataset):
    
    def __init__(self, root="/tmp/mnist/", split='train', whichFiles=[]):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        self.image_class = {}
        self.images = []
        self.targets = []
        with open(root + "../augmented_annotations.csv", "r", newline='') as a_csv:
                csv_reader = csv.DictReader(a_csv)
                for row in csv_reader:
                    self.image_class[row['image_name']] = row['type']
        for i, file in enumerate(os.listdir(root)):
            if file not in whichFiles:
                continue
            self.images.append(torch.from_numpy(cv2.imread(os.path.join(root, file))/255.))
            self.targets.append(torch.tensor(int(self.image_class[file])))
        self.classes = list(range(56))
        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_positive(self, index):
        target_class = self.targets[index].item()
        return choice(self.target2indices[target_class])

    def _sample_negative(self, index):
        target_class = self.targets[index].item()
        negative_classes = [c for c in self.classes if c != target_class and len(self.target2indices[c]) > 0]
        negative_class = choice(negative_classes)
        return choice(self.target2indices[negative_class])

    def __getitem__(self, index):
        anchor = self.images[index].permute(2,0,1).float()
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.permute(2,0,1).float(), negative.permute(2,0,1).float(), target_id

    def __len__(self):
        return len(self.images)

