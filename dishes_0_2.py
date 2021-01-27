from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()

SIZE = 224
NUMBER_OF_CLASSES = len(os.listdir("dishesDataReduced/train"))
class Net(nn.Module):
#     def __init__(self):
    def __init__(self, input_shape=(3,32,32)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7)
#         self.conv1 = nn.Conv2d(1, 32, 7)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        self.drop1 = nn.Dropout2d(0.25)
        
        # Base image object to further calculate output shape
        x = torch.randn(3, SIZE, SIZE).view(-1, 3, SIZE, SIZE)
        self._to_linear = None
        
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 128)
        
        self.drop2 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, NUMBER_OF_CLASSES)
        
    # Convolutional layers computation
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        
        # Calculate output shape of convolutional layers
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            #print('To_Linear:', self._to_linear, x[0].shape[0], x[0].shape[1], x[0].shape[2])
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = self.drop1(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
#         transforms.Resize((80, 80)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
#         transforms.Resize((80, 80)),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__': 
    
#     data_dir = 'data/hymenoptera_data'
    data_dir = 'dishesDataReduced'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    
    
    model_conv = Net().to(device)

    # Parameters of newly constructed modules have requires_grad=True by default
#     num_ftrs = model_conv.fc.in_features
# #     model_conv.fc = nn.Linear(num_ftrs, 3)
#     model_conv.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
#                                  nn.ReLU(),
#                                  nn.Dropout(0.2),
#                                  nn.Linear(512, 3),
#                                  nn.LogSoftmax(dim=1))
#     model_conv.classifier[6] = nn.Linear(4096, 2)
#     model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
#     optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
#     optimizer_conv = optim.SGD(model_conv[6].parameters(), lr=0.001, momentum=0.9)
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=10)
    
#     torch.save(model_conv, "3classModel")

    plt.ioff()
    plt.show()

