import numpy as np
import pandas as pd
from torchsummary import summary
import torch
from time import time
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from cross_validation_test.pytorchtools import EarlyStopping_kappa
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import os
from torchvision.transforms import functional as F
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler

# BASE_TRAIN_PATH = '/home/arash/Projects/Dataset/DeepDR'
BASE_TRAIN_PATH = 'E:\Dataset\DR\DeepDr'
''' Training Dataset Directory '''

# BASE_VAL_PATH = '/home/arash/Projects/Dataset/DeepDR'
# BASE_VAL_PATH = 'E:\Dataset\DR\DeepDr/regular-fundus-validation'
''' Validation Dataset Directory '''

# BASE_TEST_PATH = '/home/arash/Projects/Dataset/DeepDR/Onsite-Challenge1-2-Evaluation'
BASE_TEST_PATH = 'E:\Dataset\DR\DeepDr\Onsite-Challenge1-2-Evaluation'
''' Test Dataset Directory '''


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
        # file_name = self.train_set['image_path'][idx]
        img_id = self.train_set['image_id'][idx]
        patient_id = self.train_set['patient_id'][idx]
        file_path = os.path.join(str(patient_id),str(img_id)+'.jpg')
        label = self.train_set['patient_DR_Level'][idx]
        path = os.path.join(self.train_path , file_path)
        path = path.replace('\\', '/')
        img = Image.open(path)  # Loading Image

        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TestDataset(data.Dataset):
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
        file_name = str(self.train_set['patient_id'][idx]) + '/' + self.train_set['image_id'][idx] + '.jpg'
        label = self.train_set['patient_DRLevel'][idx]
        path = self.train_path + '/' + file_name
        path = path.replace('\\', '/')
        img = Image.open(path)  # Loading Image

        if self.transform is not None:
            img = self.transform(img)
        return img, label


batch_size = 16

params = {'batch_size': batch_size,
          'shuffle': True
          }
''' Hyper Parameters '''

learning_rate = 1e-3
''' The learning rate '''

from skimage.filters.rank import entropy
from skimage.morphology import disk


class BiChannel(object):

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be BiChanneled.

        Returns:
            PIL Image: BiChanneled image.
        """
        gray = F.to_grayscale(img, num_output_channels=1)
        # ent = entropy(np.array(gray),disk(10))
        # out = torch.cat(ent,img[1])
        return gray


transform_train = transforms.Compose([transforms.Resize((600, 600)), transforms.RandomApply([
    torchvision.transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip()], 0.7),
                                      transforms.ToTensor()])
''' Transform Images to specific size and randomly rotate and flip them '''

training_set = Dataset(os.path.join(BASE_TRAIN_PATH, 'merged_tr_vl', 'merged_tr_vl.csv'),
                       os.path.join(BASE_TRAIN_PATH, 'merged_tr_vl'),
                       transform=transform_train)

train_generator = data.DataLoader(training_set, **params)

# validation_set = Dataset(os.path.join(BASE_VAL_PATH, 'regular-fundus-validation', 'regular-fundus-validation.csv'),
#                          BASE_VAL_PATH,
#                          transform=transform_train)
''' Make a dataset of the validation set '''

# validation_generator = data.DataLoader(validation_set, **params)
''' Validation generator with the provided hyper parameters '''

test_set = TestDataset(os.path.join(BASE_TEST_PATH, 'Onsite-Challenge1-2-Evaluation_full.csv'),
                       BASE_TEST_PATH,
                       transform=transform_train)
''' Make a dataset of the test set '''

test_generator = data.DataLoader(test_set, **params)
''' Test generator with the provided hyper parameters '''

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
''' Initialize Cuda if it is available '''
print(device)

model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
''' Loaded pretrained weights for efficientnet-b3 '''
model.to(device)

print(summary(model, input_size=(3, 512, 512)))
''' Display the model structure '''

PATH_SAVE = '../Weights/'
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


epochs = 50
''' Number of epochs '''


training = True
testing = False
''' We can focus on train or testing individually '''


def quadratic_kappa(y_hat, y):
    '''
    Kappa function normally use cpu variables but with this modified version, cuda version is acceptable
    :param y_hat:
    predicted
    :param y:
    ground truth
    :return:
    quadratic kappa
    '''
    return torch.tensor(cohen_kappa_score(torch.argmax(y_hat.cpu(), 1), y.cpu(), weights='quadratic'), device='cuda:0')

torch.save(model.state_dict(), 'init.pt')

is_first = True
if training:
    # model.load_state_dict(torch.load('checkpoint.pt'))
    kfold = KFold(5, True, 1)
    cc = 0
    fold_kappa_list=[]
    for fold, (train_index, val_index) in enumerate(kfold.split(training_set)):

        # print('train: %s, test: %s' % (train_index, val_index))
        cc += 1
        print('>>>>>>>Fold '+str(cc))
        model.load_state_dict(torch.load('init.pt'))
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(val_index)
        train_loader = torch.utils.data.DataLoader(training_set,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size)
        validation_loader = torch.utils.data.DataLoader(training_set,
                                                        sampler=valid_sampler,
                                                        batch_size=batch_size)

        train_history_accuracy = []
        ''' List accuracies of all epochs on training data '''

        train_history_loss = []
        ''' List losses of all epochs on train data '''

        val_history_accuracy = []
        ''' List accuracies of all epochs on val data '''

        val_history_loss = []
        ''' List losses of all epochs on val data '''

        early_stopping = EarlyStopping_kappa(patience=5, verbose=True)
        ''' Early Stopping '''

        for epoch in range(epochs):
            if not is_first:
                '''
                Epochs loop
                '''
                running_loss = 0.0
                ''' Set the loss to zero '''

                kappa_all = 0
                count = 0
                correct = 0
                total = 0
                tr_class_correct = list(0. for _ in classes)
                tr_class_total = list(0. for _ in classes)

                pred_list = []
                label_list = []

                model.train()
                for i, data in enumerate(train_loader, 0):
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

                    print("Train Epoch : ", epoch + 1, " Batch : ", i + 1, " Loss :  ", running_loss / (i + 1),
                          " Accuracy : ",
                          accuracy,
                          " Time ", round(time() - t0, 2), "s")

                    predict_p = torch.argmax(outputs, dim=1)
                    predict_p = predict_p.detach().cpu().numpy()
                    predict = list(predict_p.astype('int'))
                    pred_list += predict
                    labels = labels.detach().cpu().numpy()
                    labels = list(labels.astype('int'))
                    label_list += labels

                kappa_all = cohen_kappa_score(pred_list, label_list, weights='quadratic')
                print(' kappa : %f' % (kappa_all))
                print('[%d epoch] Accuracy of the network on the training images: %d %%' % (
                epoch + 1, 100 * correct / total))

            correct = 0
            total = 0

            running_loss = 0.0
            vl_class_correct = list(0. for _ in classes)
            vl_class_total = list(0. for _ in classes)

            pred_list = []
            ''' make a list for prediction of all the batches'''
            label_list = []
            ''' make a list for labels of all the batches'''
            model.eval()
            ''' prepare model for evaluation '''
            with torch.no_grad():
                for i, data in enumerate(validation_loader, 0):
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

                    for j in range(labels.size(0)):
                        label = labels[j]
                        vl_class_correct[label] += c[j].item()
                        vl_class_total[label] += 1

                    running_loss += loss.item()

                    print("Validation Epoch : ", epoch + 1, " Batch : ", i + 1, " Loss :  ", running_loss / (i + 1),
                          " Accuracy : ", accuracy,
                          "Time ", round(time() - t0, 2), "s")

                    predict_p = torch.argmax(outputs, dim=1)
                    predict_p = predict_p.detach().cpu().numpy()
                    predict = list(predict_p.astype('int'))
                    pred_list += predict
                    labels = labels.detach().cpu().numpy()
                    labels = list(labels.astype('int'))
                    label_list += labels

                kappa_all = cohen_kappa_score(pred_list, label_list, weights='quadratic')
                print(' kappa : %f' % (kappa_all))

                for k in range(len(classes)):
                    if (vl_class_total[k] != 0):
                        print(
                            'Validation accuracy of %5s : %2d %%' % (
                            classes[k], 100 * vl_class_correct[k] / vl_class_total[k]))

                print('[%d epoch] Accuracy of the network on the Validation images: %d %%' % (
                epoch + 1, 100 * correct / total))


            early_stopping(kappa_all, model)

            if early_stopping.early_stop:
                print("Early stopping")
                model.load_state_dict(torch.load('checkpoint.pt'))
                early_stopping.path = 'checkpoint_fold_'+str(cc)+'.pt'
                early_stopping.save_checkpoint(early_stopping.val_kappa_max,model)
                fold_kappa_list.append(early_stopping.val_kappa_max)
                break
            is_first = False

    print('>>>>>train finished<<<<<')
    print('fold kappa list:')
    for i in range(len(fold_kappa_list)):
        print('fold '+str(i+1)+': '+str(fold_kappa_list[i]))
if testing:

    model.load_state_dict(torch.load('checkpoint.pt'))
    ''' Load the last checkpoint with the best model '''
    running_loss = 0.0
    ''' Set the loss to zero '''

    correct = 0
    total = 0

    vl_class_correct = list(0. for _ in classes)
    vl_class_total = list(0. for _ in classes)

    pred_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_generator, 0):
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

            print("Validation Epoch : ", 1 + 1, " Batch : ", i + 1, " Loss :  ", running_loss / (i + 1),
                  " Accuracy : ", accuracy,
                  "Time ", round(time() - t0, 2), "s")

            predict_p = torch.argmax(outputs, dim=1)
            predict_p = predict_p.detach().cpu().numpy()
            predict = list(predict_p.astype('int'))
            pred_list += predict
            labels = labels.detach().cpu().numpy()
            labels = list(labels.astype('int'))
            label_list += labels

        kappa_all = cohen_kappa_score(pred_list, label_list, weights='quadratic')
        print(' kappa : %f' % (kappa_all))

    for k in range(len(classes)):
        if (vl_class_total[k] != 0):
            print('Validation accuracy of %5s : %2d %%' % (classes[k], 100 * vl_class_correct[k] / vl_class_total[k]))

    print('[%d epoch] Accuracy of the network on the Validation images: %d %%' % (1 + 1, 100 * correct / total))
