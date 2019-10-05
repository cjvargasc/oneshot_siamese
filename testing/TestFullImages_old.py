import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import cv2
import os

from params.config import Config


class Tester:

    @staticmethod
    def test():

        path = "/home/mmv/siamese_test/data/qmul_logos/testing/"
        dirs = []

        for root, directories, files in os.walk(path):
            dirs = directories
            break

        net = torch.load(Config.model_path).cuda()
        net.eval()

        preprocess = transforms.Compose(
            [transforms.Resize((Config.im_w, Config.im_h)),
             transforms.ToTensor()])

        tp = 0
        fn = 0

        for dir in dirs:
            print(dir)
            directory = os.fsencode(path + dir + "/")
            cont = 0
            tests = []

            for file in os.listdir(directory):

                directory = os.fsdecode(directory)
                filename = os.fsdecode(file)

                if filename.endswith(".png"):

                    if cont == 0:
                        query = Image.open(directory + filename)
                        query = preprocess(query)
                        query.unsqueeze_(0)
                        cont += 1
                    else:
                        test = Image.open(directory + filename)
                        test = preprocess(test)
                        test.unsqueeze_(0)
                        tests.append(test)

            distances = []

            threshold = 13.404337

            # Evaluation
            with torch.no_grad():
                for test in tests:

                    x0, x1 = query, test  ##next(dataiter) ###############################

                    output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
                    euclidean_distance = F.pairwise_distance(output1, output2)

                    distances.append(euclidean_distance)
                    #print(euclidean_distance)

                    if euclidean_distance <= threshold:
                        tp += 1
                    else:
                        fn += 1

        print("tp: ", tp)
        print("fn: ", fn)
        print(str(tp / float(tp + fn)))