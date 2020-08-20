import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
import PIL
import sys
import torch
from time import time
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import os

BASE_TRAIN_PATH = 'E:\Dataset\DR\DeepDr/regular-fundus-training'
''' Training Dataset Directory '''

BASE_VAL_PATH = 'E:\Dataset\DR\DeepDr/regular-fundus-validation'
''' Validation Dataset Directory '''


class Dataset(data.Dataset):
    def __init__(self, csv_path, images_path, transform=None):
        ''' Initialise paths and transforms '''
        self.train_set = pd.read_csv(csv_path, keep_default_na=False)  # Read The CSV and create the dataframe
        self.train_path = images_path  # Images Path
        self.transform = transform  # Augmentation Transforms

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, idx):
        '''
        Receive element index, load the image from the path and transform it
        :param idx:
        Element index
        :return:
        Transformed image and its grade label
        '''
        file_name = self.train_set['image_path'][idx]
        label = self.train_set['patient_DR_Level'][idx]
        path = self.train_path + file_name
        img = Image.open(path)  # Loading Image

        if self.transform is not None:
            img = self.transform(img)
        return img, label


params = {'batch_size': 16,
          'shuffle': True
          }
''' Hyper Parameters '''

epochs = 100
''' Number of epochs '''

learning_rate = 1e-3
''' The learning rate '''

transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomApply([
                                        torchvision.transforms.RandomRotation(10),
                                        transforms.RandomHorizontalFlip()], 0.7),
                                                          transforms.ToTensor()])
''' Transform Images to specific size and randomly rotate and flip them '''

training_set = Dataset(os.path.join(BASE_TRAIN_PATH, 'regular-fundus-training', 'regular-fundus-training.csv'),
                       BASE_TRAIN_PATH,
                       transform=transform_train)
''' Make a dataset of the training set '''
training_generator = data.DataLoader(training_set, **params)
''' Train generator with the provided hyper parameters '''

validation_set = Dataset(os.path.join(BASE_VAL_PATH, 'regular-fundus-validation', 'regular-fundus-validation.csv'),
                         BASE_VAL_PATH,
                         transform=transform_train)
''' Make a dataset of the validation set '''

validation_generator = data.DataLoader(validation_set, **params)
''' Validation generator with the provided hyper parameters '''

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
''' Initialize Cuda if it is available '''
print(device)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
''' Loaded pretrained weights for efficientnet-b0 '''
model.to(device)

print(summary(model, input_size=(3, 512, 512)))
''' Display the model structure '''

PATH_SAVE = './Weights/'
''' Set the directory to record model's weights '''

if (not os.path.exists(PATH_SAVE)):
    os.mkdir(PATH_SAVE)
    ''' If the directory does not exist, it will be created there '''

criterion = nn.CrossEntropyLoss()
''' Set the loss function '''

lr_decay = 0.99
''' Set the learning rate delay '''

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
''' Set the optimizer '''

# Eye to create an 5x5 tensor
eye = torch.eye(5).to(device)
classes = [0, 1, 2, 3, 4]
''' Make a ist of classes '''

train_history_accuracy = []
''' List accuracies of all epochs on training data '''

train_history_loss = []
''' List losses of all epochs on train data '''

val_history_accuracy = []
''' List accuracies of all epochs on val data '''

val_history_loss = []
''' List losses of all epochs on val data '''

epochs = 50
''' Number of epochs '''

for epoch in range(epochs):
    '''
    Epochs loop
    '''
    running_loss = 0.0
    ''' Set the loss to zero '''

    correct = 0
    total = 0
    tr_class_correct = list(0. for _ in classes)
    tr_class_total = list(0. for _ in classes)

    vl_class_correct = list(0. for _ in classes)
    vl_class_total = list(0. for _ in classes)
    model.train()
    for i, data in enumerate(training_generator, 0):
        ''' run through batches of data from training data generator '''

        inputs, labels = data
        t0 = time()
        inputs, labels = inputs.to(device), labels.to(device)
        labels = eye[labels]
        optimizer.zero_grad()
        ''' Set the optimizer gradients to zero '''
        outputs = model(inputs)

        loss = criterion(outputs, torch.max(labels, 1)[1])
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        c = (predicted == labels.data).squeeze()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = float(correct) / float(total)

        train_history_accuracy.append(accuracy)
        train_history_loss.append(loss)

        loss.backward()
        optimizer.step()

        for j in range(labels.size(0)):
            label = labels[j]
            tr_class_correct[label] += c[j].item()
            tr_class_total[label] += 1

        running_loss += loss.item()

        print("Train Epoch : ", epoch + 1, " Batch : ", i + 1, " Loss :  ", running_loss / (i + 1), " Accuracy : ", accuracy,
              "Time ", round(time() - t0, 2), "s")

    print('[%d epoch] Accuracy of the network on the training images: %d %%' % (epoch + 1, 100 * correct / total))

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_generator, 0):
            ''' run through batches of data from validation data generator '''

            inputs, labels = data
            t0 = time()
            inputs, labels = inputs.to(device), labels.to(device)
            labels = eye[labels]
            ''' Set the optimizer gradients to zero '''
            outputs = model(inputs)

            loss = criterion(outputs, torch.max(labels, 1)[1])
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            c = (predicted == labels.data).squeeze()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            accuracy = float(correct) / float(total)
            ''' Calculate total accuracy '''

            val_history_accuracy.append(accuracy)
            val_history_loss.append(loss)

            # loss.backward()

            for j in range(labels.size(0)):
                label = labels[j]
                vl_class_correct[label] += c[j].item()
                vl_class_total[label] += 1

            running_loss += loss.item()

            print("Validation Epoch : ", epoch + 1, " Batch : ", i + 1, " Loss :  ", running_loss / (i + 1), " Accuracy : ", accuracy,
                  "Time ", round(time() - t0, 2), "s")

    for k in range(len(classes)):
        if (vl_class_total[k] != 0):
            print('Validation accuracy of %5s : %2d %%' % (classes[k], 100 * vl_class_correct[k] / vl_class_total[k]))

    print('[%d epoch] Accuracy of the network on the Validation images: %d %%' % (epoch + 1, 100 * correct / total))

    if epoch % 10 == 0 or epoch == 0:
        torch.save(model.state_dict(), os.path.join(PATH_SAVE, str(epoch + 1) + '_' + str(accuracy) + '.pth'))