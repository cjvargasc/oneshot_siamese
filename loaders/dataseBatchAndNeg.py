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

        self.epoch_index = 0  # index t pick just 1 logo in each batch

        self.random_indexes = np.random.randint(
            len(self.imageFolderDataset.imgs),
            size=int((len(self.imageFolderDataset.imgs)) / Config.train_batch_size) + 1)

    def __getitem__(self, index):

        if index == 0:  # reset the indexes every epoch
            self.random_indexes = np.random.randint(len(self.imageFolderDataset.imgs), size=int(
                (len(self.imageFolderDataset.imgs)) / Config.train_batch_size) + 1)

        img0_tuple = self.imageFolderDataset.imgs[self.random_indexes[int(index/Config.train_batch_size)]]

        while "neg_ss" in img0_tuple[0]:  # Repick random index if it is negative TODO: Not to pick negatives as anchor? experiment with this
            self.random_indexes[int(index / Config.train_batch_size)] = random.randint(0, len(self.imageFolderDataset.imgs) - 1)
            img0_tuple = self.imageFolderDataset.imgs[self.random_indexes[int(index / Config.train_batch_size)]]
            #  img0_tuple = self.imageFolderDataset.imgs[random.randint(0, len(self.imageFolderDataset.imgs))]


        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            should_get_logo = random.randint(0, 3)

            if should_get_logo > 2:

                while True:
                    # keep looping till the diff logo class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    #print(img1_tuple)
                    if img0_tuple[1] != img1_tuple[1]:
                        break
            else:

                # keep looping till a negative class image is found
                while True:
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    if "neg_ss" in img1_tuple[0]:  # TODO move fixed string to config
                        break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        ## ! channel
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
