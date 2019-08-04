__author__ = "Nitin Patil"

'''import os
import uuid
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

SEED = 42

class MNIST_Dataset(Dataset):
    """Custom Dataset for loading MNIST images"""

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
            #torch.nn.BatchNorm2d(32),
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
            #torch.nn.BatchNorm2d(64),
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
        #out = self.drop_out(out)
        out = self.linear_1(out)
        out = F.relu(out)
        logits = self.linear_2(out.view(-1, 1000))
        probas = F.log_softmax(logits, dim=1)
        return logits, probas
    
    
#################################
### Model Initialization
#################################

def preprocess_image(image):
    """
        Save and open an image and resize to 28x28
        Return numpy array
    """
    filename = 'digit' +  '__' + str(uuid.uuid1()) + '.jpg'
    with open('tmp/' + filename, 'wb') as f:
        f.write(image)
        
    # open and convert the image to grayscale
    img = Image.open('tmp/' + filename).convert(mode='L')
    
    # create a thumbnail and preserve aspect ratio
    img.thumbnail((28,28))
    return np.asarray(img)

def prediction(model, data_loader):
    test_pred = torch.LongTensor()#.to(DEVICE)
    with open("log.txt", 'w') as f:
        f.write("prediction\n")
        for features in data_loader:
            #features = features#.to(DEVICE)
            logits, probas = model(features)
            _, predicted_labels = torch.max(probas,1)
            f.write(f"logits : {logits}\n")
            f.write(f"probas : {probas}\n")
            f.write(f"predicted_labels : {predicted_labels}\n")
            test_pred = torch.cat((test_pred, predicted_labels), dim=0)

        f.write(f"test_pred : {test_pred}\n")
            
    return test_pred

def single_predict(model, image):
    image = preprocess_image(image)
    if image is None:
        return "Can't predict, when nothing is drawn"

    batch_size = 1

    with open("log.txt", 'w') as f:
        f.write("Entered single_predict\n")
    
        f.write(f"Image type : {type(image)}\n")
        
        test_dataset = MNIST_Dataset(image,
                                    transform=test_transform)

        f.write(f"test_dataset : {type(test_dataset)}\n")
        
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
                                 
        f.write(f"test_loader : {type(test_loader)}\n")
        
        test_pred = prediction(model, test_loader)
        
        f.write(f"test_pred : {test_pred}\n")
        f.write(f"test_pred Shape : {test_pred.shape}\n")
        
        return test_pred.item()
'''
