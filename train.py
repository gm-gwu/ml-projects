#!/usr/bin/env python3

# Imports here
import argparse
import matplotlib.pyplot as plt
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from PIL import Image
import os
import copy
import time
import numpy as np
import seaborn as sns
from collections import OrderedDict


def args_paser():
    parser = argparse.ArgumentParser(description='Train image classifier')
    parser.add_argument('-d', '--data_dir', type=str, default='flowers', required=False, help='Path to data_directory')
    parser.add_argument('-s', '--save_dir', type=str, default='imgcheckpoint.pth', required=False,
                        help='Set file to save checkpoints')
    parser.add_argument('-a', '--arch', type=str, default='vgg', required=False,
                        help='Set architecture to vgg or densenet, default:vgg')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, required=False, help='Set Learning Rate')
    parser.add_argument('-u', '--hidden_units', type=int, default=500, required=False, help='Set hidden_units')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Set number of epochs')
    parser.add_argument('-g', '--gpu', type=bool, required=False, default=True,
                        help='Use GPU for training, True: gpu, False: cpu')

    args = parser.parse_args()
    return args


def dataloader(data_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid', 'test']}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'valid', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    return data_transforms, image_datasets, dataloaders, dataset_sizes


def buildmodel(arch, learning_rate, gpu, hidden_units=512):
    # TODO: Build and train your network
    # If user wants to use gpu but its not available then use CPU
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if arch == "vgg":
        model = models.vgg16(pretrained=True)
        num_in_features = 25088

    elif arch == "densenet":
        model = models.densenet161(pretrained=True)
        num_in_features = 1024
    else:
        print("Invalid model name, exiting...Supported models vgg, densenet")
        exit()

    model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_in_features, hidden_units, bias=True)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(hidden_units, 128, bias=True)),
                                            ('relu2', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.5)),
                                            ('fc3', nn.Linear(128, 102, bias=True)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))

    criterion = nn.CrossEntropyLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model.to(device)
    return model, criterion, exp_lr_scheduler, optimizer, device


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs, dataset_sizes):
    since = time.time()
    train_losses, valid_losses = [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_losses.append(running_loss / dataset_sizes[phase])
            elif phase == 'valid':
                valid_losses.append(running_loss / dataset_sizes[phase])

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# TODO: Do validation on the test set
def check_accuracy_on_test(testloader, model, device):
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def savecheckpoint(chkpointname, model, image_datasets, arch, num_epochs, optimizer):
    # TODO: Save the checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx

    if arch == "vgg":
        checkpoint = {'arch': 'vgg16',
                      'classifier': model.classifier,
                      'class_to_idx': model.class_to_idx,
                      'epochs': num_epochs,
                      'state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}

    elif arch == "densenet":
        checkpoint = {'arch': 'densenet161',
                      'classifier': model.classifier,
                      'class_to_idx': model.class_to_idx,
                      'epochs': num_epochs,
                      'state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, chkpointname)


def main():
    args = args_paser()
    data_transforms, image_datasets, dataloaders, dataset_sizes = dataloader(args.data_dir)

    model, criterion, exp_lr_scheduler, optimizer, device, = buildmodel(args.arch, args.learning_rate, args.gpu, args.hidden_units)
    model_ft = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, device, args.epochs, dataset_sizes)

    check_accuracy_on_test(dataloaders['test'], model_ft, device)

    savecheckpoint(args.save_dir, model_ft, image_datasets, args.arch, args.epochs, optimizer)
    print('Train Complete')


if __name__ == '__main__':
    main()

