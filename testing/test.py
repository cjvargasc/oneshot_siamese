import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

from loaders.datasetTestFull import TestSiameseNetworkDataset  # Change this import to switch between dataset loaders
#from loaders.datasetTest import TestSiameseNetworkDataset
from params.config import Config
import math

#from models.SNalexnet import SiameseNetwork
#from models.SNtests import SiameseNetwork
from models.SNakoch import SiameseNetwork

from misc.misc import Utils

class Tester:

    @staticmethod
    def test():

        #folder_dataset_test = dset.ImageFolder(root=Config.full_test_dir)
        folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
        siamese_dataset = TestSiameseNetworkDataset(
            imageFolderDataset=folder_dataset_test,
            transform=transforms.Compose([transforms.Resize((Config.im_w, Config.im_h)), #transforms.Grayscale(num_output_channels=3),
                                           transforms.ToTensor()]), ## Grayscale: Omniglot only, check older versions
            should_invert=False)

        #net = SiameseNetwork().cuda()
        #net = torch.load(Config.best_model_path).cuda()
        net = torch.load(Config.model_path).cuda()
        net.eval()

        #  Important: the test dataset doesnt work properly with more than one thread (num_workers)
        test_dataloader = DataLoader(siamese_dataset, num_workers=1, batch_size=1, shuffle=True)

        dataiter = iter(test_dataloader)
        data_len = len(siamese_dataset)

        size_test = data_len - 1

        distances = []
        labels = []

        # Evaluation
        with torch.no_grad():
            for i in range(size_test):

                if (i % 5000 == 0):
                    print(i)  # progress

                x0, x1, label = next(dataiter)
                #concatenated = torch.cat((x0, x1), 0)
                #Utils.imshow(torchvision.utils.make_grid(concatenated))

                label = label.data.cpu().numpy()[0][0]

                output1, output2, score = net(Variable(x0).cuda(), Variable(x1).cuda())
                #euclidean_distance = F.pairwise_distance(output1, output2)
                euclidean_distance = score
                # print(euclidean_distance.cpu().data.numpy()[0])

                labels.append(label)
                distances.append(euclidean_distance)

        # TPR/FPR counting
        sc_cont = 0
        dc_cont = 0

        sc_accum = 0
        dc_accum = 0

        sc_max = -1
        sc_min = 99999

        dc_max = -1
        dc_min = 99999

        for i in range(size_test):
        # Count samples from same and different classes and finds the max / min distances
            if (labels[i] == 0):
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

        # std dev
        sc_std_accum = 0
        dc_std_accum = 0
        for i in range(size_test):
            if (labels[i] == 0):
                sc_std_accum += pow(distances[i] - sc_mean, 2)
            else:
                dc_std_accum += pow(distances[i] - dc_mean, 2)

        sc_std = math.sqrt(sc_std_accum / sc_cont)
        dc_std = math.sqrt(dc_std_accum / dc_cont)

        print("Matching_samples: ", sc_cont)
        print("Mismatching_samples: ", dc_cont)
        print(" Mean std min max")
        print("Match:", sc_mean.data.cpu().numpy()[0], sc_std, sc_min.data.cpu().numpy()[0],
              sc_max.data.cpu().numpy()[0])
        print("Non-match:", dc_mean.data.cpu().numpy()[0], dc_std, dc_min.data.cpu().numpy()[0],
              dc_max.data.cpu().numpy()[0])

        points = 20
        thresholds = []

        minn = min(sc_min, dc_min)
        maxx = max(sc_max, dc_max)

        print("min, max, step", minn.data.cpu().numpy()[0], ", ", maxx.data.cpu().numpy()[0], ", ",
              ((maxx - minn) / points).data.cpu().numpy()[0], "\n")

        for i in range(0, points + 1):
            thresholds.append((minn) + (i * ((maxx - minn) / points)))
            # print((min) + (i * ((max - min) / points)))

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

        print(thresholds_str)
        print(tp_str)
        print(tn_str)
        print(fp_str)
        print(fn_str)
        print(TPR_str)
        print(FPR_str)
