"""
GOAL:
Load images from disk → convert to tensors

WHY:
Model cannot read jpg files directly
"""
import os
import cv2
import torch
from torch.utils.data import Dataset


class FaceDataset:

    def __init__(self, data_dir):
        self.paths = []
        self.labels = []
        for file in os.listdir(data_dir):
            if file.endswith('.jpg'):
                label = int(file.split("_")[0])
                self.paths.append(os.path.join(data_dir, file))
                self.labels.append(label)
                


    def __len__(self):
        return(len(self.paths))


    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).permute(2,0,1).float() / 255

        return img, self.labels[idx]
