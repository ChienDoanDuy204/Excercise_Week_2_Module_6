import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split
import numpy as np


def Code_decode_label(data_dir):
    classes = os.listdir(data_dir)
    label2idx = {label:idx for idx, label in enumerate(classes)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label

def get_imgPath_label(data_dir, label2idx):
    img_paths = []
    labels = []
    for clss in os.listdir(data_dir):
        for img  in os.listdir(os.path.join(data_dir, clss)):
            img_paths.append(os.path.join(data_dir,clss,img))
            labels.append(label2idx[clss])
    return img_paths, labels


class ImageDataSet(Dataset):
    def __init__(self, img_paths:list = None, labels: list = None, train: bool = True, val_split : float = 0.0, random_state: int = 14, transforms :list = None):
        super().__init__()
        self.img_paths = img_paths
        self.labels = labels
        self.val_split = val_split
        self.random_state = random_state
        self.train = train
        self.split_train_val()
        self.transforms = transforms
        self.transformer = Compose(self.transforms)
    def split_train_val(self):
        if self.val_split:
            train_data, val_data = train_test_split(list(zip(self.img_paths, self.labels)), test_size=self.val_split, shuffle=True, random_state=self.random_state, stratify=self.labels)
            if self.train:
                self.img_paths, self.labels = zip(*train_data)
            if not self.train:
                self.img_paths, self.labels = zip(*val_data)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        img = self.img_paths[index]
        label = self.labels[index]
        img = read_image(img).float()/255.0
        if self.transforms is not None:
            img = self.transformer(img)
        return img, label

