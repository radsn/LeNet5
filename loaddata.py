import torch
from torchvision import datasets
import numpy as np


# define data pipeline based on prebuilt functions
class MyDataset(datasets.MNIST):
    def __init__(self, *args, noise_range=(-5, 5), **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_range = noise_range

    def __getitem__(self, index):

        img, target = self.data[index], int(self.targets[index])
        img = img.float()

        # create a random noise map
        noise = torch.from_numpy(np.random.uniform(low=self.noise_range[0], high=self.noise_range[1], size=img.shape)).float()
        # add it to image
        img = noise + img

        # add a channel dimension
        img = img[None, :, :]

        # standardization
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

        # return a data bundle
        return img, target
