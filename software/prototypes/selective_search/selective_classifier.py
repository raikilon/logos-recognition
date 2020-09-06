from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cv2
from scipy.cluster.vq import *
import os
import logging
import datetime
import pickle
import sys
import random
import string


class SelectiveSearch(object):

    def __init__(self, load=False):
        # ----------------- SETTINGS -----------------
        self.iou = 0.0
        self.test_threshold = 0.5
        self.test_bounding_boxes = 500
        self.hard_negatives_bounding_boxes = 100
        # ----------------- END SETTINGS -----------------

        # log initialization
        logging.basicConfig(filename='data.log', filemode='w', level=logging.INFO)
        # variables initialization
        self.images_dir = "train/"
        # load the histograms
        self.histograms = np.loadtxt(self.images_dir + "/histograms.out")
        # load classes from a txt
        self.names = [line.rstrip('\n') for line in open(self.images_dir + "classes.txt")]
        # load vocabulary
        self.vocabulary = np.loadtxt(self.images_dir + "/voc.out")
        # load standard scaler
        self.std_slr = pickle.load(open(self.images_dir + "/std_slr.out", "rb"))
        # instantiate array of x element (each for a logo type)
        self.images_class = [None] * len(self.names)
        self.classifiers = [None] * len(self.names)
        self.probabilities = [0] * len(self.names)
        self.results = []

        logging.warning("Start classifiers training ...")
        # loop over the different classes (Mastercard vs other, etc.)
        for i, dir_name in enumerate(self.names):
            # if the classifier is already present load it otherwise create it
            if load:
                self.classifiers[i] = pickle.load(open(self.images_dir + dir_name + ".out", "rb"))
            else:
                self.images_class[i] = np.loadtxt(self.images_dir + "classes_" + dir_name + ".out")
                # use random forest with 1000 trees
                self.classifiers[i] = RandomForestClassifier(1000)
                self.classifiers[i].fit(self.histograms, self.images_class[i])
                pickle.dump(self.classifiers[i], open(self.images_dir + dir_name + ".out", "wb"))
        logging.warning("END.")

    def classify(self, img, print_img=False):
        """
        Classify and generate probabilities
        :param img: image to classify
        :param print_img: True to save each image that gave a important probability
        :return: nothing
        """
        probs = []
        # loop over the different logos classes to compute the histogram
        for i in range(len(self.names)):
            hist = self.compute_hist(img, i)
            if hist is not None:
                # Compute probability that the histogram is the logo i
                prob = self.classifiers[i].predict_proba(hist)[0]
                # Save probability and logo type
                probs.append((prob[0], i))
        # Sort the probabilities to have the highest as the first element
        probs.sort(key=lambda tup: tup[0], reverse=True)
        # loop over all probabilities (one for each class) and take the highest until now
        for i in range(len(probs)):
            # if prob is higher than the probability until now replace it
            if probs[i][0] > self.probabilities[probs[i][1]]:
                self.probabilities[probs[i][1]] = probs[i][0]
                # print image with random name
                if print_img:
                    # name compose of the class and 10 random characters
                    name = self.names[probs[i][1]] + "-" + ''.join(
                        random.choices(string.ascii_uppercase + string.digits, k=10))
                    cv2.imwrite("img/" + name + ".png", img)
                    logging.info("The class is " + str(probs[i][1]) + " - " + str(probs[i][0]) + " - " + name)
                break

    def OpenCV(self, img):
        """
        Selective search with OpenCV on the given images and save the results
        (it is bugged with some type of image https://github.com/opencv/opencv_contrib/issues/705)
        :param img: image to process
        :return: nothing (logging the probabilities)
        """
        # ti ve szre reset probabilities and parameters
        self.probabilities = [0] * len(self.names)
        self.results = []
        # create Selective Search Segmentation Object using default parameters
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        # set input image on which we will run segmentation
        ss.setBaseImage(img)

        ss.switchToSelectiveSearchQuality()

        # run selective search segmentation on input image
        rects = ss.process()
        # number of region proposals to show

        # iterate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if i < self.test_bounding_boxes:
                x, y, w, h = rect
                self.classify(img[y:y + h, x:x + w], False)
            else:
                break
        # log and save all the probabilities
        for i in range(len(self.probabilities)):
            self.results.append((self.names[i], self.probabilities[i]))
            logging.info(
                self.names[i] + " is present at : %.1f %%" % (self.probabilities[i] * 100))
        logging.info("_______________________")

    def compute_hist(self, img, i):
        """
        Compute histogram with sift descriptors and vocabulary
        :param img: image to compute histogram
        :param i: class number
        :return: generated histogram
        """
        # Construct SIFT object
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute SIFT
        kp, descriptors = sift.detectAndCompute(gray, None)
        # if descriptor are > 0 and < 200 compute histogram
        if descriptors is not None and len(descriptors) < 200:
            histograms = np.zeros((1, 10000))
            # Assign codes from a code book to observations.
            words, distance = vq(descriptors, self.vocabulary)
            for w in words:
                histograms[0][w] += 1
            # Standardize
            histograms = self.std_slr.transform(histograms)
        else:
            histograms = None
        return histograms

    def update_classifier(self, file_name, banner_dir):
        """
        Update the classifier with hard negatives
        :param file_name: file with ground-truth bounding boxes (class order same as directory order)
        :param banner_dir: banner images dir
        :return: nothing (it saves a new classifier)
        """
        # load file lines
        lines = [line.rstrip('\n') for line in open(file_name)]
        # loop over the different classes (Visa, MasterCard, etc.)
        for i in range(len(self.names)):
            # histograms for fit
            histograms = []
            # classes for fit
            classes = []
            #  search for the start of data for the i-th class
            start = lines.index(str(self.names[i])) + 1
            # if it is last class end is last element
            if i == len(self.names) - 1:
                end = len(lines) - 1
            # if it is not the last class go until the element before the next class
            else:
                end = lines.index(str(self.names[i + 1]), start)
            # loop over the different images
            for k, dir_name in enumerate(os.listdir(banner_dir)):
                boxes = []
                # load all data for class i
                for j in range(start, end):
                    # load only boxes for current image
                    if int(lines[j].split("_")[0]) == k:
                        # loading bounding box
                        boxes.append(list(map(int, lines[j].split("_")[1].split(","))))
                # add border to the image (cv2 bug)
                bordersize = 200
                border = cv2.copyMakeBorder(cv2.imread(os.path.join(banner_dir, dir_name)), top=bordersize,
                                            bottom=bordersize, left=bordersize, right=bordersize,
                                            borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                # set input image on which we will run segmentation
                ss.setBaseImage(border)
                ss.switchToSelectiveSearchQuality()
                # run selective search segmentation on input image
                rects = ss.process()
                # iterate over all the region proposals
                for rect_num, rect in enumerate(rects):
                    # loop only over 200 bounding boxes
                    if rect_num < self.hard_negatives_bounding_boxes:
                        x, y, w, h = rect
                        # IoU check
                        check = False
                        # iterate over loaded boxes
                        for box in boxes:
                            # check if IoU between loaded boxes and selective search boxes is over 50%
                            if self.bb_intersection_over_union(box, [x, y, x + w, y + h]) > self.iou:
                                check = True
                                break
                        # Compute bag of word histogram
                        hist = self.compute_hist(border[y:y + h, x:x + w], i)
                        if hist is not None:
                            # If IoU is below the 50% and the classifier predict that it is the processed class
                            # train the classifier that it said no
                            if not check and self.classifiers[i].predict(hist) == 0:
                                classes.append(1)
                                histograms.append(hist[0])
                    else:
                        break
            if len(classes) != 0:
                # Fit classifier with new data
                self.classifiers[i].fit(np.append(np.array(histograms), self.histograms, axis=0),
                                        np.append(np.array(classes), self.images_class[i], axis=0))
                # Save new classifier
                pickle.dump(self.classifiers[i], open(self.images_dir + self.names[i] + ".out", "wb"))

    @staticmethod
    def bb_intersection_over_union(box_a, box_b):
        """
        Compute intersection over union
        see https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        :param box_a: first bounding box
        :param box_b: first bounding box 
        :return: percentage of IoU
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        # max is to avoid those corner cases where the rectangles are not overlapping,
        # but the intersection area still computes to be greater than 0
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        boxBArea = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        try:
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou
        except ZeroDivisionError:
            return 0
        # return the intersection over union value

    def test(self, banner_dir, others_dir, file_name):
        """
        Test the selective search with OpenCV
        :param banner_dir: images with logos
        :param others_dir: images without logos
        :param file_name: file that contain which images contains which logo in banner_dir
        :return: nothing (log the quality measures of the confusion matrix)
        """
        # variable initialization
        tp = 0
        tn = 0
        fn = 0
        fp = 0

        # border size because OpenCV generate bounding box outside the image border (bug)
        bordersize = 200
        # load logos presence from file
        lines = [line.rstrip('\n') for line in open(file_name)]

        # loop over all images in banner_dir
        for i, dir_name in enumerate(os.listdir(banner_dir)):

            # add white border to the processed image
            border = cv2.copyMakeBorder(cv2.imread(os.path.join(banner_dir, dir_name)), top=bordersize,
                                        bottom=bordersize, left=bordersize, right=bordersize,
                                        borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

            self.OpenCV(border)
            # compute true positives, false negatives, etc.
            for k in range(len(self.results)):
                if self.results[k][0] in lines[i]:
                    if self.results[k][1] > self.test_threshold:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if self.results[k][1] > self.test_threshold:
                        fp += 1
                    else:
                        tn += 1
        # loop over all image in others_dir
        for i, dir_name in enumerate(os.listdir(others_dir)):
            border = cv2.copyMakeBorder(cv2.imread(os.path.join(others_dir, dir_name)), top=bordersize,
                                        bottom=bordersize, left=bordersize, right=bordersize,
                                        borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
            self.OpenCV(border)
            for k in range(len(self.results)):
                if self.results[k][1] > self.test_threshold:
                    fp += 1
                else:
                    tn += 1

        # generate quality measures of the confusion matrix
        try:
            # temp values because can happen division by zero
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            acc = (tp + tn) / (tp + tn + fp + fn)
            logging.info("Sensitivity: %.1f" % (tpr * 100))
            logging.info("Specificity: %.1f" % (tnr * 100))
            logging.info("Precision: %.1f" % (ppv * 100))
            logging.info("Negative Predictive Value: %.1f" % (npv * 100))
            logging.info("Accuracy: %.1f" % (acc * 100))
        except ZeroDivisionError:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logging.error(
                "Division by zero at line: " + str(exc_tb.tb_lineno))


def main():
    # ----------------- TEST -----------------
    start_time = datetime.datetime.now()
    classifier = SelectiveSearch(False)
    logging.info("Initialisation time: " + str(datetime.datetime.now() - start_time))
    start_time = datetime.datetime.now()
    classifier.test("C:/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/banner/logos/",
                    "C:/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/banner/other/",
                    "values.txt")
    logging.info("Time: " + str(datetime.datetime.now() - start_time))

    # ----------------- HARD NEGATIVE UPDATE -----------------
    # start_time = datetime.datetime.now()
    # classifier = SelectiveSearch(False)
    # logging.info("Initialisation time: " + str(datetime.datetime.now() - start_time))
    # start_time = datetime.datetime.now()
    # classifier.update_classifier("boundingbox.txt",
    #                             "C:/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/banner/logos/")
    # logging.info("Time: " + str(datetime.datetime.now() - start_time))

    # ----------------- TEST AFTER HARD NEGATIVE UPDATE -----------------
    # start_time = datetime.datetime.now()
    # classifier = SelectiveSearch(True)
    # logging.info("Initialisation time: " + str(datetime.datetime.now() - start_time))
    # start_time = datetime.datetime.now()
    # classifier.test("C:/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/banner/logos/",
    #                "C:/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/banner/other/",
    #                "values.txt")
    # logging.info("Time: " + str(datetime.datetime.now() - start_time))


if __name__ == "__main__":
    main()
