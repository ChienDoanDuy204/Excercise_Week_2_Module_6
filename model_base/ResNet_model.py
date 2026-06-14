import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
from torchinfo import summary
from tqdm import tqdm
from MLP import BaseMLP, ResidualBlock
from sklearn.model_selection import train_test_split

class ResidualNetworkBase(BaseMLP):
    def __init__(self):
        super().__init__()
    def predict(self, X):
        with torch.no_grad():
            self.model.eval()
            X = X.to(self.device)
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)
    def compute_loss(self, logits, y):
        return self.criterion(logits,y)
    def get_accuracy(self, logits, y):
        try:
            return torch.mean((torch.argmax(logits,dim =1) == y).float())
        except:
            return torch.mean((torch.argmax(logits,dim =1) == torch.argmax(y, dim = 1)).float())


class ResNet18(ResidualNetworkBase):
    def __init__(self, num_classes:int = 1000):
        super().__init__()

############ ------- Built architecture ResNET-18 ------- ################       
        # block1
        block1 = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        self.Add_layer(block1)

        # Residual block1
        block1_sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', stride = 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(num_features=64),
        )
        downsample1 = None
        residual_block1 = ResidualBlock(block=block1_sequential,downsample=downsample1)
        self.Add_layer([residual_block1, nn.ReLU(inplace=True)])

        # Residual block2
        block2_sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', stride = 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(num_features=64),
        )
        downsample2 = None
        residual_block2 = ResidualBlock(block=block2_sequential,downsample=downsample2)
        self.Add_layer([residual_block2, nn.ReLU(inplace=True)])

        # Projection shortcut 3
        block3_sequential = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(num_features=128),
        )
        downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride = 2),
            nn.BatchNorm2d(num_features=128)
        )
        residual_block3 = ResidualBlock(block=block3_sequential,downsample=downsample3)
        self.Add_layer([residual_block3, nn.ReLU(inplace=True)])

        # Residual block 4
        block4_sequential = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', stride = 1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(num_features=128),
        )
        downsample4 = None
        residual_block4 = ResidualBlock(block=block4_sequential,downsample=downsample4)
        self.Add_layer([residual_block4, nn.ReLU(inplace=True)])

        # project shortcut 5
        block5_sequential = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=256)
        )
        downsample5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256 ,kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=256)
        )
        residual_block5 = ResidualBlock(block=block5_sequential, downsample= downsample5)
        self.Add_layer([residual_block5,nn.ReLU(inplace=True)])

        # residual block 6
        block6_sequential = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=256)
        )
        downsample6 = None
        residual_block6 = ResidualBlock(block=block6_sequential, downsample= downsample6)
        self.Add_layer([residual_block6,nn.ReLU(inplace=True)])


        # project shortcut 7
        block7_sequential = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=512)
        )
        downsample7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512 ,kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=512)
        )
        residual_block7 = ResidualBlock(block=block7_sequential, downsample= downsample7)
        self.Add_layer([residual_block7,nn.ReLU(inplace=True)])

        # residual block 8
        block8_sequential = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=512)
        )
        downsample8 = None
        residual_block8 = ResidualBlock(block=block8_sequential, downsample= downsample8)
        self.Add_layer([residual_block8,nn.ReLU(inplace=True)])

        # average pooling
        block9 = [
            nn.AvgPool2d(kernel_size=7),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes)
        ]
        self.Add_layer(block9)