import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader

from loaddata import MyDataset
from model import LeNet

# batch size: should not be too small or too large
BATCH_SIZE = 32

# data augmentation
NOISE_RANGE = (0, 0)

# folders for loading and saving models
LOAD_FROM = '/mnist/test'

# single device training/testing. Modification needs for enabling multiple devices.
# use single GPU
device = torch.device("cuda:0")
# use single CPU
# device = torch.device("cpu")


def main():

    # build dataset: where and how one slice of data is prepared
    my_dataset = MyDataset(
        root=LOAD_FROM,
        train=False,
        transform=None,
        download=True,
        noise_range=NOISE_RANGE,
    )

    # build dataloader: how data are distributed (batch size, shuffling, how to divide dataset, how to distribute for multuple devices, etc.)
    test_loader = DataLoader(
        dataset=my_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # initialize the model
    model = LeNet()
    # fetch model to the device, and switch to train mode
    model = model.to(device).eval()

    # stat
    correct = 0

    # iterate all data in the same epoch
    # no grad to save computation and memory space
    with torch.no_grad():
        for data, target in test_loader:

            # data: [BATCH_SIZE, 1, H, W]
            # target: [BATCH_SIZE]
            # print(data.shape)
            # print(target.shape)

            # put dat to the device
            data = data.to(device)
            target = target.to(device)

            # run the forward function
            predictions = model(x=data, is_test=True)
            predictions = np.argmax(predictions.cpu().numpy(), axis=1)

            correct += np.equal(target.cpu().numpy(), predictions).sum()

    print(100.0 * correct / len(my_dataset))


if __name__ == '__main__':
    main()
