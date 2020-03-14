import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

#from loaders.datasetTestFull import TestSiameseNetworkDataset  # Change this import to switch between dataset loaders
from loaders.datasetTests import TestSiameseNetworkDataset
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
        #net = torch.load(Config.model_path).cuda()
        net = torch.load("/home/mmv/Documents/2.projects/SiamesePaper/trained_models/alex/ft/testmodel197").cuda()
        net.eval()

        #  Important: the test dataset doesnt work properly with more than one thread (num_workers)
        test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=False)

        dataiter = iter(test_dataloader)
        data_len = len(siamese_dataset)

        size_test = data_len - 1

        distances = []
        labels = []

        # mAP variables
        current_cls = torch.tensor([0])
        cls_scores = []
        cls_gts = []
        current_cls_score = torch.FloatTensor().cuda()
        current_cls_gts = torch.FloatTensor().cuda()

        # Evaluation
        with torch.no_grad():
            for i in range(size_test):

                if (i % 5000 == 0):
                    print(i)  # progress

                x0, x1, label, current = next(dataiter)
                #concatenated = torch.cat((x0, x1), 0)
                #Utils.imshow(torchvision.utils.make_grid(concatenated))

                # mAP: store previous scores and reset
                if current_cls.data.cpu().numpy()[0] != current.data.cpu().numpy()[0]:
                    cls_scores.append(current_cls_score)
                    cls_gts.append(current_cls_gts)
                    current_cls_score = torch.FloatTensor().cuda()
                    current_cls_gts = torch.FloatTensor().cuda()
                    current_cls = current

                label = label.data.cpu().numpy()[0][0]

                output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
                euclidean_distance = torch.sigmoid(F.pairwise_distance(output1, output2))
                # euclidean_distance = score
                # print(euclidean_distance.cpu().data.numpy()[0])

                labels.append(label)
                distances.append(euclidean_distance)

                # mAP: store predictions and ious
                current_cls_score = torch.cat((current_cls_score, euclidean_distance), 0)
                current_cls_gts = torch.cat((current_cls_gts, torch.FloatTensor([label]).cuda()), 0)

        # Store last class
        cls_scores.append(current_cls_score)
        cls_gts.append(current_cls_gts)

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
        mAP_str = "mAP: "

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

            mAP = Tester.calc_mAP(cls_scores, cls_gts, thresh)
            mAP_str += str(mAP.data.cpu().numpy()) + " "

        print(thresholds_str)
        print(tp_str)
        print(tn_str)
        print(fp_str)
        print(fn_str)
        print(TPR_str)
        print(FPR_str)
        print(mAP_str)

    @staticmethod
    def calc_mAP(cls_scores, gts, threshold):

        """ Debug
        # First class follows the example from:
        # https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
        threshold = 0.5
        cls_scores = [torch.FloatTensor([0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), torch.FloatTensor([0.2, 0.3, 0.4, 0.4, 0.6])]
        gts = [torch.FloatTensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 0]), torch.FloatTensor([0, 0, 0, 1, 0])]
        """

        APs = torch.zeros(len(cls_scores))

        for cls_idx in range(len(cls_scores)):
            precisions = []
            recalls = []

            # order both tensors by confidence score
            cls_scores[cls_idx], indices = torch.sort(cls_scores[cls_idx])
            gts[cls_idx] = torch.zeros_like(gts[cls_idx]).scatter_(dim=0, index=indices, src=gts[cls_idx])

            # Keep detections only
            mask = cls_scores[cls_idx] <= threshold
            cls_scores[cls_idx] = cls_scores[cls_idx][mask]
            gts[cls_idx] = gts[cls_idx][mask]

            tp_seen = 0
            tot_tps = (gts[cls_idx] == 0).sum(0)

            for score_idx in range(cls_scores[cls_idx].size()[0]):

                if gts[cls_idx][score_idx].data.cpu().numpy() == 0:
                    tp_seen += 1

                # proportion of tps (tps_seen / rows_seen)
                precisions.append(tp_seen / (float)(score_idx + 1))

                # Div 0 error if theres not at least 1 tp in class
                if tot_tps.data.cpu() != 0:
                    recalls.append(tp_seen / (float)(tot_tps.data.cpu()))  # proportion of tps over possible positives
                else:
                    recalls.append(0)

            current_recall = 0
            max_pr = 0
            acum = 0
            for r in range(len(recalls)):

                if precisions[r] > max_pr:
                    max_pr = precisions[r]

                if recalls[r] != current_recall:
                    acum += (recalls[r] - current_recall) * max_pr
                    current_recall = recalls[r]
                    max_pr = 0
            APs[cls_idx] = acum

        mAP = APs.sum(0) / (float)(APs.data.size(0))
        return mAP.data.cpu()
