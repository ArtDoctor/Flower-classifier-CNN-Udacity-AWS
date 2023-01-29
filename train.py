import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
from torch import nn
import torch.optim as optim

from utils import model_utils, train_logging


parser = argparse.ArgumentParser(description='Model training parameters.')
parser.add_argument('directory', metavar='dir', type=str,
                    help='directory with images for training')
parser.add_argument('--architecture', metavar='arch', type=str, choices=['vgg16', 'resnet50'],
                    nargs="?", default='vgg16', help='model architecture on which CNN based')
parser.add_argument('--learning_rate', metavar='lr', type=float, nargs="?",
                    default=0.01, help='hyperparameter: learning rate')
parser.add_argument('--hidden_units', metavar='hu', type=int, nargs="?",
                    default=4096, help='hyperparameter: amount of hidden units in NN')
parser.add_argument('--train_on_gpu', action='store_true',
                    default=False, help='defines if training is on gpu or not')
parser.add_argument('--batch_size', metavar='batch', type=int, nargs="?",
                    default=20, help='batch size for training')

# Parse all args
args = parser.parse_args()

data_dir = args.directory
batch_size = args.batch_size
learning_rate = args.learning_rate
hidden_size = args.hidden_units
model_architecture = args.architecture
using_cuda = args.train_on_gpu

# Load data
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
training_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(p=0.5)

])

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=training_data_transforms)
test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=data_transforms)
val_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=data_transforms)
print('Num training images: ', len(train_dataset))
print('Num test images: ', len(test_dataset))
print('Num validation images: ', len(val_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Define model
model, criterion, optimizer = model_utils.define_model(hidden_size, model_architecture, learning_rate)

# Training model
print("Started Training")
n_epochs = 2

for epoch in range(1, n_epochs + 1):
    for batch_i, (data, target) in enumerate(train_loader):
        # Train model
        model.train()
        if using_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_i % batch_size == batch_size - 1:
            print('Epoch %d, Batch %d train acc: %.3f, train loss: %.3f, val acc: %.3f, val loss: %.3f' %
                  (epoch, batch_i + 1,
                   train_logging.calc_accuracy(model, train_loader, using_cuda),
                   train_logging.calc_loss(model, train_loader, criterion, using_cuda),
                   train_logging.calc_accuracy(model, val_loader, using_cuda),
                   train_logging.calc_loss(model, val_loader, criterion, using_cuda)))

# Save model in path of current date so that model will not be
# overwritten for sure
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.feedforward.parameters(), lr=0.01)

model.class_to_idx = train_dataset.class_to_idx
model_path = 'model_ ' + datetime.now().strftime("%d/%m/%Y_%H:%M:%S") + '.pth'
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': learning_rate,
            'hidden_size': hidden_size,
            'architecture': model_architecture,
            }, model_path)
