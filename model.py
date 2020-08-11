import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        # extra parameters for making the model extendable
        super(LeNet, self).__init__()
        # initialized parameters
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )
        self.fc1 = nn.Linear(
            in_features=120,
            out_features=84,
            bias=True
        )
        self.fc2 = nn.Linear(
            in_features=84,
            out_features=10,
            bias=True
        )
        # self-defined initialization
        self.init_weights()

    def init_weights(self):
        # module initialization: in case you wish to use non-default initialization methods.
        for m in self.parameters():
            nn.init.normal_(m, mean=0., std=0.1)

    def forward(self, x, is_test):
        # default function to run when call y = Model(x)
        if is_test:
            x = self.forward_test(x)
        else:
            x = self.forward_train(x)

        return x

    def forward_test(self, x):
        # forward propagation for testing
        # padding 28 -> 32
        x = F.pad(input=x, pad=[2, 2, 2, 2], mode='constant', value=0)

        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(
            x,
            kernel_size=2,
            stride=2
        )
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(
            x,
            kernel_size=2,
            stride=2
        )
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.shape[0], 120)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)

        return x

    def forward_train(self, x):
        # forward propagation for training

        # padding 28 -> 32
        x = F.pad(input=x, pad=[2, 2, 2, 2], mode='constant', value=0)

        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(
            x,
            kernel_size=2,
            stride=2
        )
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(
            x,
            kernel_size=2,
            stride=2
        )
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.shape[0], 120)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)

        return F.log_softmax(input=x, dim=1)



