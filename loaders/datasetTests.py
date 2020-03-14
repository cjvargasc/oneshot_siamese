import numpy as np
import random
import torch
import PIL.ImageOps
from torch.utils.data import Dataset
from PIL import Image


class TestSiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

        #random.seed(123)

    def __getitem__(self, index):

        # Pick the same image twice
        img0_tuple = self.imageFolderDataset.imgs[index // 2]

        # once with the logo and once without
        should_get_same_class = index % 2

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
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), img0_tuple[1]

    def __len__(self):
        #return 10 # Debug only
        return len(self.imageFolderDataset.imgs) * 2