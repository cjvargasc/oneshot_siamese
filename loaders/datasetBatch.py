import numpy as np
import random
import torch
import PIL.ImageOps
from torch.utils.data import Dataset
from params.config import Config
from PIL import Image


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

        np.random.seed(123)
        random.seed(123)

        # Defines a set of random logos by index (for each epoch)
        self.random_indexes = np.random.randint(
            len(self.imageFolderDataset.imgs),
            size=int((len(self.imageFolderDataset.imgs)) / Config.train_batch_size) + 1)

    def __getitem__(self, index):

        # reset the indexes every epoch
        if index == 0:
            self.random_indexes = np.random.randint(len(self.imageFolderDataset.imgs), size=int(
                (len(self.imageFolderDataset.imgs)) / Config.train_batch_size) + 1)

        # get the index for the current batch
        img0_tuple = self.imageFolderDataset.imgs[self.random_indexes[int(index/Config.train_batch_size)]]

        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)

        # Search for class by looping random indexes
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        #img0 = img0.convert("L")
        #img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
