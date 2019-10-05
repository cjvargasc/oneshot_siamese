from skimage import img_as_float
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET

from selectivesearch import selectiveSearch


class Sampler:

    @staticmethod
    def get_neg_samples():

        directory = "/home/mmv/Documents/3.datasets/openlogo/Annotations/"
        save_dir = "/home/mmv/Documents/3.datasets/neg_ss/"
        candidates_to_save = 3

        for filename in os.listdir(directory):
            if filename.endswith(".xml"):

                root = ET.parse(directory + filename).getroot()
                im_file = root.find("filename").text

                object_el = root.find("object")

                bbox_el = object_el.find("bndbox")
                xmin = int(bbox_el.find("xmin").text)
                ymin = int(bbox_el.find("ymin").text)
                xmax = int(bbox_el.find("xmax").text)
                ymax = int(bbox_el.find("ymax").text)

                bbox_area = (xmax - xmin) * (ymax - ymin)

                im_test = cv2.imread("/home/mmv/Documents/3.datasets/openlogo/JPEGImages/" + im_file)

                print("ss starting")

                test_im = np.copy(im_test)
                height, width, channels = im_test.shape
                ratio = 500 / width
                test_im = cv2.resize(test_im, (500, int(ratio * height)))
                test_im = test_im[:, :, ::-1]
                test_im = img_as_float(test_im)

                img_lbl, regions = selectiveSearch.selective_search(
                    test_im, scale=100.0, sigma=0.8, min_size=50)  # scale= 70 or 60 sigma= 0.2 or 0.3 respectively
                # alexnet 80 0.8 50

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

                    # distance between two rects
                    inters_area = Sampler.intersection_area(x, y, x+w, y+h, xmin, ymin, xmax, ymax)

                    if inters_area <= (bbox_area * 0.05):
                        # crop
                        crop_img = im_test[y:y + h, x:x + w]
                        cv2.imwrite(save_dir + im_file.split(".")[0] + str(detections) + ".png", crop_img)
                        detections += 1
                        if detections >= candidates_to_save:
                            break

    @staticmethod
    def intersection_area(x1, y1, x1b, y1b, x2, y2, x2b, y2b):  # returns None if rectangles don't intersect
        dx = min(x1b, x2b) - max(x1, x2)
        dy = min(y1b, y2b) - max(y1, y2)
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0


if __name__ == "__main__":
    Sampler.get_neg_samples()
