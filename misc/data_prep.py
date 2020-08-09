import os
import shutil
import cv2
import xml.etree.ElementTree as ET

if __name__ == "__main__":
    """ Utils file to crop and split the openlogo dataset """

    test_classes = [
        "3m",
        "abus",
        "accenture",
        "adidas",
        "adidas1",
        "adidas_text",
        "airhawk",
        "airness",
        "aldi",
        "allianz",
        "amcrest",
        "americanexpress_text",
        "android",
        "anz",
        "anz_text",
        "apc",
        "apecase",
        "armitron",
        "aspirin",
        "asus",
        "athalon",
        "audi",
        "audi_text",
        "axa",
        "basf",
        "bionade",
        "blackmores",
        "blizzardentertainment",
        "bmw",
        "boeing",
        "boeing_text",
        "bosch",
        "bosch_text",
        "budweiser_text",
        "burgerking_text",
        "canon",
        "carglass",
        "carlsberg",
        "carters",
        "cartier",
        "caterpillar",
        "chanel",
        "costa",
        "costco",
        "cvs",
        "cvspharmacy",
        "danone",
        "dexia",
        "dhl",
        "fritos",
        "gap",
        "generalelectric",
        "gillette",
        "goodyear",
        "google",
        "gucci",
        "hp",
        "hsbc",
        "hsbc_text",
        "huawei",
        "huawei_text",
        "hyundai",
        "hyundai_text",
        "ibm",
        "jello",
        "kraft",
        "lacoste",
        "lacoste_text",
        "lamborghini",
        "lego",
        "levis",
        "lexus",
        "nasa",
        "nb",
        "nescafe",
        "netflix",
        "nike",
        "nintendo",
        "quick",
        "rbc",
        "recycling",
        "redbull",
        "redbull_text",
        "reebok",
        "reebok1",
        "shell",
        "shell_text",
        "shell_text1",
        "siemens",
        "singha",
        "skechers",
        "sony",
        "soundcloud",
        "t-mobile",
        "tnt",
        "tommyhilfiger",
        "tostitos",
        "total",
        "toyota",
        "toyota_text",
        "tsingtao",
        "vaio",
        "velveeta",
        "verizon",
        "visa",
        "vodafone",
        "volkswagen",
        "volkswagen_text",
        "volvo",
        "walmart",
        "walmart_text",
        "warnerbros",
        "wellsfargo",
        "wellsfargo_text",
        "wii",
        "williamhill",
        "windows",
        "wordpress",
        "xbox",
        "yahoo",
        "yamaha",
        "yonex",
        "yonex_text",
        "youtube",
        "zara"
    ]

    openlogo_path = "/home/mmv/Documents/3.datasets/openlogo/"
    train_dir = "/home/mmv/Documents/3.datasets/openlogo/test_split/train/"
    test_dir = "/home/mmv/Documents/3.datasets/openlogo/test_split/test/"
    annotations_directory = openlogo_path + "Annotations/"
    dataset_directory = openlogo_path + "JPEGImages/"

    for filename in os.listdir(annotations_directory):
        if filename.endswith(".xml"):

            root = ET.parse(annotations_directory + filename).getroot()
            im_file = root.find("filename").text

            object_el = root.find("object")

            bbox_el = object_el.find("bndbox")
            xmin = int(bbox_el.find("xmin").text)
            ymin = int(bbox_el.find("ymin").text)
            xmax = int(bbox_el.find("xmax").text)
            ymax = int(bbox_el.find("ymax").text)

            ob_class = object_el.find("name").text

            im = cv2.imread(dataset_directory + im_file)

            if ob_class in test_classes:
                # create folder if doesnt exists
                if not os.path.exists(test_dir + ob_class):
                    os.mkdir(test_dir + ob_class)

                crop_img = im[ymin:ymax, xmin:xmax]
                cv2.imwrite(test_dir + ob_class + "/" + im_file.split(".")[0] + ".png",
                            crop_img)

            else:
                # create folder if doesnt exists
                if not os.path.exists(train_dir + ob_class):
                    os.mkdir(train_dir + ob_class)

                crop_img = im[ymin:ymax, xmin:xmax]
                cv2.imwrite(train_dir + ob_class + "/" + im_file.split(".")[0] + ".png",
                            crop_img)
