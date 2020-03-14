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
        self.index = 0

        self.class_indexes = []  # stores the index to the first image of each class
        self.current_class_index = 0
        self.class_indexes.append(0)

        np.random.seed(123)
        random.seed(123)

        aux_index = 0
        for sample in range(len(self.imageFolderDataset.imgs)):
            if self.imageFolderDataset.imgs[sample][1] > aux_index:  # if next class
                self.class_indexes.append(sample)
                aux_index += 1

        self.test_classes = aux_index

        self.leng = int(len(self.imageFolderDataset.imgs) * 3) - len(self.class_indexes)
        print(self.leng)
        print(len(self.class_indexes))
        print(int(self.leng / (len(self.class_indexes))))

    def __getitem__(self, idx):

        current = self.class_indexes[self.current_class_index]

        if self.index >= int(self.leng / len(self.class_indexes)) + 5: # len(self.imageFolderDataset.imgs):  # if index > dataset size
            # if go to next class
            self.current_class_index += 1
            print("current class: ", self.current_class_index)
            current = self.class_indexes[self.current_class_index]
            self.index = 0

        if self.index == current:
            self.index += 1

        img0_tuple = self.imageFolderDataset.imgs[current]

        ###
        if current + self.index >= len(self.imageFolderDataset.imgs) - 1:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        else:
            img1_tuple = self.imageFolderDataset.imgs[current + self.index + 1]

            if img0_tuple[1] != img1_tuple[1]: # If it
                while True:
                    # keep looping till diff class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    if img0_tuple[1] != img1_tuple[1]:
                        break

        ###

        self.index += 1

        #print(current)
        #print(self.index)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        ## 1 channel
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
        return self.leng
