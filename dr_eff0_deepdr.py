import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Installing Libraries
# !pip install efficientnet-pytorch
# !pip install torchsummary
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
# Input dataset is APTOS2019

import os


def metricsCompute(predict, label, labels=[0, 1, 2, 3, 4]):
    ap = []
    recall = []
    f1 = []
    for p in labels:
        fake_label = (label == p)
        fake_predict = (predict == p)

        ap.append(precision_score(fake_label.ravel(), fake_predict.ravel()))
        recall.append(recall_score(fake_label.ravel(), fake_predict.ravel()))
        f1.append(f1_score(fake_label.ravel(), fake_predict.ravel()))

    kappa = cohen_kappa_score(predict, label, weights='quadratic')

    return np.array(ap), np.array(recall), np.array(f1), kappa


# Load Datasets
BASE_TRAIN_PATH = 'E:\Dataset\DR\DeepDr/regular-fundus-training'
# BASE_TRAIN_PATH = 'E:\Dataset\DR\DeepDr/regular-fundus-training/regular-fundus-training'
BASE_VAL_PATH = 'E:\Dataset\DR\DeepDr/regular-fundus-validation/regular-fundus-validation'
# train_dataset = pd.read_csv(os.path.join(BASE_TRAIN_PATH, 'regular-fundus-training.csv'))
# final_test_dataset = pd.read_csv(os.path.join(BASE_VAL_PATH, 'regular-fundus-validation.csv'))
#
# train_dataset.head(3)
# final_test_dataset.head(3)


class Dataset(data.Dataset):
    def __init__(self, csv_path, images_path, transform=None):
        self.train_set = pd.read_csv(csv_path,keep_default_na=False)  # Read The CSV and create the dataframe
        self.train_path = images_path  # Images Path
        self.transform = transform  # Augmentation Transforms

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, idx):
        file_name = self.train_set['image_path'][idx]
        label = self.train_set['patient_DR_Level'][idx]
        path = self.train_path+ file_name
        img = Image.open(path)  # Loading Image
        # img = Image.open(file_name)  # Loading Image
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# Hyper Parameters
params = {'batch_size': 16,
          'shuffle': True
          }
epochs = 100
learning_rate = 1e-3

transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomApply([
    torchvision.transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip()], 0.7),
                                      transforms.ToTensor()])

training_set = Dataset(os.path.join(BASE_TRAIN_PATH, 'regular-fundus-training', 'regular-fundus-training.csv'), BASE_TRAIN_PATH,
                       transform=transform_train)
training_generator = data.DataLoader(training_set, **params)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)

model.to(device)

print(summary(model, input_size=(3, 512, 512)))

PATH_SAVE = './Weights/'
if (not os.path.exists(PATH_SAVE)):
    os.mkdir(PATH_SAVE)

criterion = nn.CrossEntropyLoss()
lr_decay = 0.99
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Eye to create an 5x5 tensor
eye = torch.eye(5).to(device)
classes = [0, 1, 2, 3, 4]

history_accuracy = []
history_loss = []
epochs = 50

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = list(0. for _ in classes)
    class_total = list(0. for _ in classes)
    for i, data in enumerate(training_generator, 0):
        inputs, labels = data
        t0 = time()
        inputs, labels = inputs.to(device), labels.to(device)
        labels = eye[labels]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        c = (predicted == labels.data).squeeze()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = float(correct) / float(total)

        history_accuracy.append(accuracy)
        history_loss.append(loss)

        loss.backward()
        optimizer.step()

        for j in range(labels.size(0)):
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1

        running_loss += loss.item()

        print("Epoch : ", epoch + 1, " Batch : ", i + 1, " Loss :  ", running_loss / (i + 1), " Accuracy : ", accuracy,
              "Time ", round(time() - t0, 2), "s")
    for k in range(len(classes)):
        if (class_total[k] != 0):
            print('Accuracy of %5s : %2d %%' % (classes[k], 100 * class_correct[k] / class_total[k]))

    print('[%d epoch] Accuracy of the network on the Training images: %d %%' % (epoch + 1, 100 * correct / total))

    if epoch % 10 == 0 or epoch == 0:
        torch.save(model.state_dict(), os.path.join(PATH_SAVE, str(epoch + 1) + '_' + str(accuracy) + '.pth'))

torch.save(model.state_dict(), os.path.join(PATH_SAVE, 'Last_epoch' + str(accuracy) + '.pth'))

#
# plt.plot(history_accuracy)
# plt.plot(history_loss)


# model.load_state_dict(torch.load('./Weights/41_0.9729655925723648.pth'))
#
# model.eval()

# test_transforms = transforms.Compose([transforms.Resize(512),
#                                       transforms.ToTensor(),])

# def predict_image(image):
#     image_tensor = test_transforms(image)
#     image_tensor = image_tensor.unsqueeze_(0)
#     input = Variable(image_tensor)
#     input = input.to(device)
#     output = model(input)
#     index = output.data.cpu().numpy().argmax()
#     return index

# submission=pd.read_csv(BASE_PATH+'sample_submission.csv')
#
# submission.head(3)
#
# submission_csv=pd.DataFrame(columns=['id_code','diagnosis'])
#
# IMG_TEST_PATH=os.path.join(BASE_PATH,'test_images/')
# for i in range(len(submission)):
#     img=Image.open(IMG_TEST_PATH+submission.iloc[i][0]+'.png')
#     label = int(submission.iloc[i][1])
#     prediction=predict_image(img)
#     submission_csv=submission_csv.append({'id_code': submission.iloc[i][0],'diagnosis': prediction, 'label':label},ignore_index=True)
#     if(i%10==0 or i==len(submission)-1):
#         print('[',32*'=','>] ',round((i+1)*100/len(submission),2),' % Complete')
#
# submission_csv.to_csv('submission.csv',index=False)
