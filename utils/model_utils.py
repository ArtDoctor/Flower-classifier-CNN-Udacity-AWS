import torch
from torch import nn
import torch.optim as optim
import torchvision


# Define the feedforward neural network
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define the full model by combining VGG16 and the feedforward network
class FullModel(torch.nn.Module):
    def __init__(self, vgg16, feedforward):
        super(FullModel, self).__init__()
        self.vgg16 = vgg16.features
        self.feedforward = feedforward

    def forward(self, x):
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)
        x = self.feedforward(x)
        return x


def define_model(hidden_size, model_architecture, learning_rate):
    # Create an instance of the full model
    input_size = 7 * 7 * 512
    output_size = 102

    if model_architecture == 'vgg16':
        # Load VGG
        pre_model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        for param in pre_model.parameters():
            param.requires_grad = False
    elif model_architecture == 'resnet50':
        # Load Resnet
        pre_model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        for param in pre_model.parameters():
            param.requires_grad = False
    else:
        raise Exception("No model architecture found, please choose vgg or resnet")

    # Main Model
    feedforward = Feedforward(input_size, hidden_size, output_size)
    model = FullModel(pre_model, feedforward)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.feedforward.parameters(), lr=learning_rate)
    return model, criterion, optimizer
