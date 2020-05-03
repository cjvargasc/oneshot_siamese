import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import torchvision

import torch.nn.functional as F

from lossfunction.contrastive import ContrastiveLoss
from loaders.datasetBatch import SiameseNetworkDataset
from models import SNresnet18, SNalexnet, SNdenseNet, SNinception, SNsqueeze, SNvgg, SNakoch, SNtests
from misc.misc import Utils
from params.config import Config


class Trainer:

    @staticmethod
    def train():

        print("Training process initialized...")
        print("dataset: ", Config.training_dir)

        folder_dataset = dset.ImageFolder(root=Config.training_dir)

        siamese_dataset = SiameseNetworkDataset(
            imageFolderDataset=folder_dataset,
            transform=transforms.Compose([transforms.Resize((Config.im_w, Config.im_h)),
                                                                              transforms.ToTensor()]),
            should_invert=False)

        train_dataloader = DataLoader(siamese_dataset,
                                      shuffle=False,
                                      num_workers=8,
                                      batch_size=Config.train_batch_size)

        print("lr:     ", Config.lrate)
        print("batch:  ", Config.train_batch_size)
        print("epochs: ", Config.train_number_epochs)

        net = Trainer.selectModel()  # Model defined in config

        net.train()

        criterion = ContrastiveLoss()

        optimizer = optim.SGD(net.net_parameters, lr=Config.lrate)

        counter = []
        loss_history = []

        best_loss = 10**15
        best_epoch = 0
        break_counter = 0  # break after 20 epochs without loss improvement

        # for epoch in range(0, Config.train_number_epochs):
        for epoch in range(0, Config.train_number_epochs):

            average_epoch_loss = 0
            for i, data in enumerate(train_dataloader, 0):

                img0, img1, label = data
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

                # Debug: Training data visualization
                #concatenated = torch.cat((data[0], data[1]), 0)
                #Utils.imshow(torchvision.utils.make_grid(concatenated))

                if Config.distanceLayer:
                    scores = net(img0, img1)
                else:
                    output1, output2 = net(img0, img1)
                    scores = F.pairwise_distance(output1, output2)

                optimizer.zero_grad()

                if Config.bceLoss:
                    loss = F.binary_cross_entropy_with_logits(scores, label)
                else: # contrastive
                    loss = criterion(scores, label)

                loss.backward()

                optimizer.step()

                average_epoch_loss += loss.item()

            average_epoch_loss = average_epoch_loss / i

            print("Epoch number {}\n Current loss {}\n".format(epoch, average_epoch_loss))
            counter.append(epoch)
            loss_history.append(average_epoch_loss)

            if epoch % 50 == 0:
                torch.save(net, Config.best_model_path + str(epoch))

            if average_epoch_loss < best_loss:
                best_loss = average_epoch_loss
                best_epoch = epoch
                torch.save(net, Config.best_model_path)
                print("------------------------Best epoch: ", epoch)
                break_counter = 0

            if break_counter >= 20:
                print("Training break...")
                #break

            break_counter += 1

        torch.save(net, Config.model_path)

        print("best: ", best_epoch)
        Utils.show_plot(counter, loss_history)

    @staticmethod
    def selectModel():

        if Config.model == "resnet":
            return SNresnet18.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).cuda()
        elif Config.model == "alexnet":
            return SNalexnet.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).cuda()
        elif Config.model == "dense":
            return SNdenseNet.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).cuda()
        elif Config.model == "inception":
            return SNinception.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).cuda()
        elif Config.model == "vgg":
            return SNvgg.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).cuda()
        elif Config.model == "squeeze":
            return SNsqueeze.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).cuda()
        elif Config.model == "koch":
            return SNakoch.SiameseNetwork(lastLayer=Config.distanceLayer, pretrained=Config.pretrained).cuda()
        elif Config.model == "tests":
            return SNtests.SiameseNetwork(pretrained=Config.pretrained).cuda()

