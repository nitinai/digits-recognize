__author__ = "Nitin Patil"

import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class MNIST_Dataset(Dataset):
    "Custom Dataset for loading MNIST images"

    def __init__(self, img, transform=None):
        self.X = img.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
        self.y = None
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            x = self.transform(self.X[index])
        
        if self.y is not None:
            return x, self.y[index]
        
        return x

    def __len__(self):
        return self.X.shape[0]
        
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class CNN(torch.nn.Module):
    """Adapted from:
    https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/dl-competition/winning-solution.ipynb
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,
                         stride=2,
                        padding=0))  
        self.bn1    = torch.nn.BatchNorm2d(32)
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,
                         stride=2,
                        padding=0))
        self.bn2    = torch.nn.BatchNorm2d(64)
        self.drop_out = torch.nn.Dropout()
        self.linear_1 = torch.nn.Linear(2304, 1000)
        self.bn_fc    = torch.nn.BatchNorm2d(1000)
        self.linear_2 = torch.nn.Linear(1000,num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear_1(out)
        out = F.relu(out)
        logits = self.linear_2(out.view(-1, 1000))
        probas = F.log_softmax(logits, dim=1)
        return logits, probas
    
    
def preprocess_image(image):
    """
        Save and open an image and resize to 28x28
        Return numpy array
    """
    # open and convert the image to grayscale
    img = image.convert(mode='L')
    
    # create a thumbnail and preserve aspect ratio
    img.thumbnail((28,28))
    return np.asarray(img)

def prediction(model, data_loader):
    test_pred = torch.LongTensor()
    
    for features in data_loader:
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas,1)
        test_pred = torch.cat((test_pred, predicted_labels), dim=0)

    return test_pred

def single_predict(model, image):
    image = preprocess_image(image)
    if image is None:
        return "Can't predict, when nothing is drawn"

    test_dataset = MNIST_Dataset(image, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False)
                                
    test_pred = prediction(model, test_loader)
    return test_pred.item()
