from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.task_3.covid_classifier import CovidNet
from src.task_3.pytorch_classifier import CNN

# Adapted from https://medium.com/@aaysbt/fashion-mnist-data-training-using-pytorch-7f6ad71e96f4
# mostly used for nice output formatting
def train_model(model, train, valid, epochs, filename):
    # Defines the loss function and optimiser
    criterion = nn.NLLLoss()
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    # Starts the loss at infinite so the weight is always updated first time
    valid_loss_min = np.Inf
    # Prepares the model to be trained
    model.train()
    train_losses, valid_losses = [], []
    for e in range(0, epochs):
        running_loss = 0
        valid_loss = 0
        # train the model
        for images, labels in train:
            optimiser.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * images.size(0)

        for images, labels in valid:
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            valid_loss += loss.item() * images.size(0)

        running_loss = running_loss / len(train.sampler)
        valid_loss = valid_loss / len(valid.sampler)
        train_losses.append(running_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(e + 1, running_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('validation loss decreased({:.6f} --> {:.6f}). Saving Model ...'.format(
                valid_loss_min, valid_loss))
            torch.save(model.state_dict(), filename)
            valid_loss_min = valid_loss
    return train_losses, valid_losses

# Adapted from https://medium.com/@aaysbt/fashion-mnist-data-training-using-pytorch-7f6ad71e96f4
# mostly used for nice output formatting
def model_test(model, test):
    test_loss = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    criterion = nn.NLLLoss()

    model.eval()
    for images, labels in test:
        # forward pass
        output = model(images)
        # calculate loss
        loss = criterion(output, labels)
        # update the test loss
        test_loss += loss.item() * images.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to labels
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
        # calculate test accuracy
        for i in range(len(labels)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    # calculate test loss
    test_loss = test_loss / len(test.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' %
                  (str(i), 100 * class_correct[i] / class_total[i],
                   np.sum(class_correct[i]), np.sum(class_total[i])))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)
    ))

def load_fashion():
    # Loads the Fashion MNIST dataset using pytorch, used from code at:
    # https://medium.com/@aaysbt/fashion-mnist-data-training-using-pytorch-7f6ad71e96f4
    # Converts the labels to one hot format
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,), )])

    train_set = datasets.FashionMNIST('~/pytorch/F_MNIST_data', download=True, train=True, transform=transform)
    test_set = datasets.FashionMNIST('~/pytorch/F_MNIST_data', download=True, train=False, transform=transform)

    ind = list(range(len(train_set)))
    np.random.shuffle(ind)
    split = int(np.floor(0.2 * len(train_set)))
    train_sample = SubsetRandomSampler(ind[:split])
    validation_sample = SubsetRandomSampler(ind[split:])

    train_loader = DataLoader(train_set, sampler=train_sample, batch_size=64)
    validation_loader = DataLoader(train_set, sampler=validation_sample, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    return train_loader, validation_loader, test_loader

def load_covid():
    # loads the covid dataset and resizes the images to be
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.ToTensor()
    ])
    train_set = datasets.ImageFolder(root='../../data/Covid19-dataset/train', transform=transform)
    test_set = datasets.ImageFolder(root='../../data/Covid19-dataset/test', transform=transform)

    ind = list(range(len(train_set)))
    np.random.shuffle(ind)
    split = int(np.floor(0.2 * len(train_set)))
    train_sample = SubsetRandomSampler(ind[:split])
    validation_sample = SubsetRandomSampler(ind[split:])

    train_loader = DataLoader(train_set, sampler=train_sample, batch_size=32)
    validation_loader = DataLoader(train_set, sampler=validation_sample, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    return train_loader, validation_loader, test_loader, train_set.classes


if __name__ == "__main__":
    # # Loads the cnn
    # model = CNN(0)
    # model.load_state_dict(torch.load('models/model.pt'))
    # model_test(model, test_loader)

    # # Loads the fashion-MNIST dataset
    # train_loader, validation_loader, test_loader = load_fashion

    # # Creates an instance of the network
    # cnn = CNN(0.2)

    # Loads the covid database and generates the labels from which folder the items were in
    train_loader, validation_loader, test_loader, classes = load_covid()
    # # Creates an instance of the network
    # cnn = CovidNet()
    #
    # # Trains the cnn
    # train_losses, valid_losses = train_model(cnn, train_loader, validation_loader, 40, 'models/model_covid.pt')
    #
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(valid_losses, label='Valid Loss')
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss Value")
    # plt.legend()
    # plt.show()

    # Loads covid model
    model = CovidNet()
    model.load_state_dict(torch.load('models/model_covid.pt'))
    model_test(model, test_loader)
    print(classes)
