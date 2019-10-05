import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
from skimage import img_as_float
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET

from models.SNalexnet import SiameseNetwork
from selectivesearch import selectiveSearch
from params.config import Config


class Tester:

    @staticmethod
    def test():

        # TODO: move to config
        fxy = 2
        #
        preprocess = transforms.Compose(
            [transforms.Resize((Config.im_w, Config.im_h)),
             transforms.ToTensor()])

        query_class_name = "3m"
        im_query = cv2.imread("/home/mmv/siamese_test/data/qmul_logos/training/3m/3m1.png")
        query = cv2.cvtColor(im_query, cv2.COLOR_BGR2RGB)
        query = Image.fromarray(query)
        query = preprocess(query)
        query.unsqueeze_(0)

        # net = torch.load(Config.model_path).cuda()
        net = SiameseNetwork().cuda()
        net.eval()



        threshold = 50

        print("the test has begun")

        directory = "/home/mmv/Documents/camilo/datasets/openlogo/Annotations/"
        for filename in os.listdir(directory):
            if filename.endswith(".xml"):

                root = ET.parse(directory + filename).getroot()
                im_file = root.find("filename").text

                object_el = root.find("object")
                class_name = object_el.find("name").text

                bbox_el = object_el.find("bndbox")
                xmin = bbox_el.find("xmin").text
                ymin = bbox_el.find("ymin").text
                xmax = bbox_el.find("xmax").text
                ymax = bbox_el.find("ymax").text

                if class_name == "3m":

                    im_test = cv2.imread("/home/mmv/Documents/camilo/datasets/openlogo/JPEGImages/" + im_file)

                    print("ss starting")

                    test_im = np.copy(im_test)
                    height, width, channels = im_test.shape
                    ratio = 500 / width
                    test_im = cv2.resize(test_im, (500, int(ratio * height)))
                    test_im = test_im[:, :, ::-1]
                    test_im = img_as_float(test_im)

                    img_lbl, regions = selectiveSearch.selective_search(
                        test_im, scale=100.0, sigma=0.8, min_size=50)  # scale= 70 or 60 sigma= 0.2 or 0.3 respectively
                    #alexnet 80 0.8 50

                    print("ss done")

                    candidates = set()
                    for r in regions:
                        # excluding same rectangle (with different segments)
                        if r['rect'] in candidates:
                            continue
                        # excluding regions smaller than 2000 pixels
                        if r['size'] < 300:
                            continue
                        # distorted rects
                        x, y, w, h = r['rect']
                        if h == 0 or w == 0:
                            continue
                        if w / h > 3 or h / w > 3:
                            continue
                        candidates.add(r['rect'])

                    print("Region evaluation")
                    detections = 0
                    for x, y, w, h in candidates:

                        x = int(x / ratio)
                        y = int(y / ratio)
                        w = int(w / ratio)
                        h = int(h / ratio)

                        # crop
                        crop_img = im_test[y:y + h, x:x + w]
                        # print(x, y, w, h)

                        # convert to PIL
                        test = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        test = Image.fromarray(test)
                        test = preprocess(test)
                        test.unsqueeze_(0)

                        # evaluate
                        with torch.no_grad():

                            x0, x1 = query, test  ##next(dataiter) ###############################

                            output1, output2 = net(Variable(x0).cuda(), Variable(x1).cuda())
                            euclidean_distance = F.pairwise_distance(output1, output2)

                            #distances.append(euclidean_distance)
                            #print(euclidean_distance.data.cpu().numpy()[0])

                            if euclidean_distance.data.cpu().numpy()[0] <= threshold:
                                detections += 1
                                #print("detection found in: ", x, y, w, h)
                                cv2.rectangle(im_test, (x, y), (x+w, y+h), (0, 0, 255), 2)

                    print("iterations: ", len(candidates))
                    print("detections: ", detections)

                    height, width, channels = im_test.shape
                    if width > 900:
                        im_test = cv2.resize(im_test, (900, int((900 / width) * height)))
                    cv2.imshow("test", im_test)
                    cv2.waitKey()
                    torch.cuda.empty_cache()
                    print("end for")

'''

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

'''