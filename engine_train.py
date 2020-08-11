import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from loaddata import MyDataset
from model import LeNet

# batch size: should not be too small or too large
BATCH_SIZE = 32

# data augmentation
NOISE_RANGE = (-5, 5)

# total epoches
MAX_EPOCH = 50

# folders for loading and saving models
SAVE_TO = '/mnist/test'
LOAD_FROM = None

# optimizer parameters
LR = 2e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# log interval (in the unit of epoch)
LOG_INTERVAL = 2

# single device training/testing. Modification needs for enabling multiple devices.
# use single GPU
device = torch.device("cuda:0")
# use single CPU
# device = torch.device("cpu")


def main():

    # build dataset: where and how one slice of data is prepared
    my_dataset = MyDataset(
        root=SAVE_TO,
        train=True,
        transform=None,
        download=True,
        noise_range=NOISE_RANGE,
    )

    # build dataloader: how data are distributed (batch size, shuffling, how to divide dataset, how to distribute for multuple devices, etc.)
    train_loader = DataLoader(
        dataset=my_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # initialize the model
    model = LeNet()
    # fetch model to the device, and switch to train mode
    model = model.to(device).train()

    # define optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    # define loss. log softmax(logits) + NLLLoss
    # equals to: softmax(logits) + cross entropy loss
    criterion = nn.NLLLoss()

    # for each epoch
    for epoch in range(0, MAX_EPOCH):
        # iterate all data in the same epoch
        for data, target in train_loader:

            # data: [BATCH_SIZE, 1, H, W]
            # target: [BATCH_SIZE]
            # print(data.shape)
            # print(target.shape)

            # put dat to the device
            data = data.to(device)
            target = target.to(device)

            # run the forward function
            predictions = model(x=data, is_test=False)
            loss = criterion(input=predictions, target=target)

            print(loss)

            # remove previous values
            optimizer.zero_grad()
            # back-propagation for steps
            loss.backward()
            # update weights
            optimizer.step()

        if epoch % LOG_INTERVAL == 0:
            torch.save(model.state_dict(), '{}/model_{}.pth'.format(SAVE_TO, epoch))
            torch.save(optimizer.state_dict(), '{}/optimizer_{}.pth'.format(SAVE_TO, epoch))


if __name__ == '__main__':
    main()
