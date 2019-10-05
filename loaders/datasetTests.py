import numpy as np
import torch
import PIL.ImageOps
from torch.utils.data import Dataset
from PIL import Image


class TestSiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
        self.index = 0

        self.class_indexes = []  # stores the index to the first image of each class
        self.current_class_index = 0
        self.class_indexes.append(0)

        aux_index = 0
        for sample in range(len(self.imageFolderDataset.imgs)):
            if self.imageFolderDataset.imgs[sample][1] > aux_index:  # if next class
                self.class_indexes.append(sample)
                aux_index += 1

        self.test_classes = aux_index

    def __getitem__(self, idx):

        current = self.class_indexes[self.current_class_index]

        if self.index >= len(self.imageFolderDataset.imgs):  # if index > dataset size
            # if go to next class
            self.current_class_index += 1
            current = self.class_indexes[self.current_class_index]
            self.index = 0

        if self.index == current:
            self.index += 1

        img0_tuple = self.imageFolderDataset.imgs[current]
        img1_tuple = self.imageFolderDataset.imgs[self.index]
        self.index += 1

        #print(current)
        #print(self.index)

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

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return (len(self.imageFolderDataset.imgs) * len(self.class_indexes)) - len(self.class_indexes)
