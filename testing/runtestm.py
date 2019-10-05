import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import math

from torchvision import models

def init_weights(m):
    #for param in model_conv.parameters():
    if type(m) == nn.Linear:
        #torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(random.random())
        m.weight.data.fill_(random.random())

##########################################################

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


#########################################################

class Config():
    training_dir = "/home/camilo/siamese_test/data/qmul_logos/training/"
    testing_dir = "/home/camilo/siamese_test/data/qmul_logos/testing/"
    #training_dir = "/home/camilo/siamese_test/data/logos/training/"
    #testing_dir = "/home/camilo/siamese_test/data/logos/testing/"
    train_batch_size = 64#32
    train_number_epochs = 50


#########################################################
class TestSiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
        self.index = 1

    def __getitem__(self, idx):
        #img0_tuple = random.choice(self.imageFolderDataset.imgs)

        img0_tuple = self.imageFolderDataset.imgs[0]
        img1_tuple = self.imageFolderDataset.imgs[self.index]
        self.index += 1

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

class TestAllSiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
        self.index = 0

        self.class_indexes = [] # stores the index to the first image of each class
        self.current_class_index = 0
        self.class_indexes.append(0)

        aux_index = 0
        for sample in range(len(self.imageFolderDataset.imgs)):
            if self.imageFolderDataset.imgs[sample][1] > aux_index: # if next class
                self.class_indexes.append(sample)
                aux_index += 1

        self.test_classes = aux_index

    def __getitem__(self, idx):
        #img0_tuple = random.choice(self.imageFolderDataset.imgs)

        current = self.class_indexes[self.current_class_index]

        if self.index >= len(self.imageFolderDataset.imgs): # if index > dataset size
            # if go to next class
            self.current_class_index += 1
            current = self.class_indexes[self.current_class_index]
            self.index = 0

        if self.index == current:
            self.index += 1

        img0_tuple = self.imageFolderDataset.imgs[current]
        img1_tuple = self.imageFolderDataset.imgs[self.index]
        self.index += 1

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
        return (len(self.imageFolderDataset.imgs) * len(self.class_indexes)) - len(self.class_indexes)


####################################################################################

class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

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

####################################################################################

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.model_conv = torchvision.models.resnet18(pretrained=True)
        self.net_parameters = []

        for name, child in self.model_conv.named_children():
            if name != "fc":
                for name2, params in child.named_parameters():
                    params.requires_grad = False
            else:
                for name2, params in child.named_parameters():
                    self.net_parameters.append(params)


        # Parameters of newly constructed modules have requires_grad=True by default
        #num_ftrs = self.model_conv.fc.in_features

        #self.model_conv.fc = nn.Linear(num_ftrs, 20)
        #self.model_conv.fc.requires_grad = True

        self.model_conv = self.model_conv.cuda()


    def forward_once(self, x):
        #print(self.model_conv)
        output = self.model_conv(x)
        #output = output.view(output.size()[0], -1)
        #print(output.size())
        #output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        #print(input1)############################################################################## normalize -1,1
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

############################################################################


folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = TestAllSiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((200, 200)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        , should_invert=False)

#net = SiameseNetwork().cuda()

net = torch.load(model_name).cuda()

net.eval()

test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True) # num workers = 6
dataiter = iter(test_dataloader)

data_len = len(siamese_dataset)
print(data_len)

size_test = data_len - 1

distances = []
labels = []

'''
for i in range(20):
    _, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),
           'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0]))
'''

with torch.no_grad():
    for i in range(size_test):

        if(i % 100 == 0):
            print(i)

        x0, x1, label = next(dataiter)
        # concatenated = torch.cat((x0, x1), 0)

        label = label.data.cpu().numpy()[0][0]
        #print(label)

        output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
        euclidean_distance = F.pairwise_distance(output1, output2)
        #print(euclidean_distance.cpu().data.numpy()[0])
        #imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0]))

        labels.append(label)
        distances.append(euclidean_distance)


#############################################################################################################


sc_cont = 0
dc_cont = 0

sc_accum = 0
dc_accum = 0

sc_max = -1
sc_min = 99999

dc_max = -1
dc_min = 99999

for i in range(size_test):

    if(labels[i] == 0):
        sc_cont += 1
        sc_accum += distances[i]

        if distances[i] < sc_min:
            sc_min = distances[i]
        if distances[i] > sc_max:
            sc_max = distances[i]
    else:
        dc_cont += 1
        dc_accum += distances[i]

        if distances[i] < dc_min:
            dc_min = distances[i]
        if distances[i] > dc_max:
            dc_max = distances[i]

sc_mean = sc_accum / sc_cont
dc_mean = dc_accum / dc_cont

sc_std_accum = 0
dc_std_accum = 0
for i in range(size_test):
    if (labels[i] == 0):
        sc_std_accum += pow(distances[i] - sc_mean, 2)
    else:
        dc_std_accum += pow(distances[i] - dc_mean, 2)

sc_std = math.sqrt(sc_std_accum / sc_cont)
dc_std = math.sqrt(dc_std_accum / dc_cont)

print("Same class samples: ", sc_cont)
print("Diff class samples: ", dc_cont)
print("Mean std min max")
print("Match:", sc_mean.data.cpu().numpy()[0], sc_std, sc_min.data.cpu().numpy()[0], sc_max.data.cpu().numpy()[0])
print("Non-match:", dc_mean.data.cpu().numpy()[0], dc_std, dc_min.data.cpu().numpy()[0], dc_max.data.cpu().numpy()[0])

#thresholds = [0.20, 0.23, 0.26, 0.3, 0.34, 0.36, 0.40, 0.42]

points = 20
thresholds = []
min = min(sc_min, dc_min)
max = max(sc_max, dc_max)

print("min, max, step", min.data.cpu().numpy()[0], ", ", max.data.cpu().numpy()[0], ", ", ((max - min) / points).data.cpu().numpy()[0], "\n")

for i in range(0, points+1):
    thresholds.append((min) + (i * ((max - min) / points)))
    #print((min) + (i * ((max - min) / points)))

thresholds_str = "threshold: "
tp_str = "tp: "
tn_str = "tn: "
fp_str = "fp: "
fn_str = "fn: "
TPR_str = "TPR: "
FPR_str = "FPR: "

for thresh in thresholds:

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(size_test):

        if (labels[i] == 0 and distances[i] < thresh):
            tp += 1
        elif (labels[i] == 0 and distances[i] > thresh):
            fn += 1
        elif (labels[i] == 1 and distances[i] < thresh):
            fp += 1
        elif (labels[i] == 1 and distances[i] > thresh):
            tn += 1

    thresholds_str += str(thresh.data.cpu().numpy()[0]) + " "
    tp_str += str(tp) + " "
    tn_str += str(tn) + " "
    fp_str += str(fp) + " "
    fn_str += str(fn) + " "
    TPR_str += str(tp / float(tp + fn)) + " "
    FPR_str += str(fp / float(fp + tn)) + " "

    #print("threshold: ", thresh.data.cpu().numpy()[0])
    #print("tp: ", tp)
    #print("tn: ", tn)
    #print("fp: ", fp)
    #print("fn: ", fn)
    #print("TPR: ", tp / float(tp + fn))
    #print("FPR: ", fp / float(fp + tn))

print(thresholds_str)
print(tp_str)
print(tn_str)
print(fp_str)
print(fn_str)
print(TPR_str)
print(FPR_str)