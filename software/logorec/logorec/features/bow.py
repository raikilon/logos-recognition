from lib.feature import Feature
import os
import shutil
import cv2
import numpy as np
import sklearn.preprocessing as sk
import pickle
from scipy.cluster.vq import *


class Bow(Feature):
    """
    In computer vision, the bag-of-words model (BoW model) can be applied to image classification, by treating image
    features as words. In document classification, a bag of words is a sparse vector of occurrence counts of words;
    that is, a sparse histogram over the vocabulary. In computer vision, a bag of visual words is a vector of occurrence
    counts of a vocabulary of local image features.
    """

    def __init__(self):
        """
        Initiate the feature by creating the BoW directory in data/feature and initialize other important parameters.
        """
        # data directory
        self.feature_directory = 'data/features/bow'
        # Hard negative bounding box
        self.bounding_box_file = 'data/features/boundingbox.txt'
        # Hard negative files
        self.banner_dir = 'data/features/banner/'
        # Generate feature bow directory if it does not exist
        if not os.path.exists(self.feature_directory):
            os.makedirs(self.feature_directory)

    def services(self, data, classifier):
        """
        Compute the probability of the different services contained in the data using the given classifier.

        :param data: List of image to process
        :param classifier: Classifier implementation
        :return: List of probability of the different services (one for each category)
        """
        values = self.__generate_values(data, classifier)
        max_value = 0
        # Search max index in probabilities (number of categories)
        for val in values:
            if val[0] > max_value:
                max_value = val[0]
        # Add element to the array (start at 0)
        max_value += 1
        # List of probabilities for each category
        probabilities = [0] * max_value
        for val in values:
            if (val[1]*100) > probabilities[val[0]]:
                probabilities[val[0]] = (100*val[1])

        return probabilities

    def probability(self, data, classifier):
        """
        Compute probability that the given data is a web shop using the given classifier. If files for the
        classification are missing, raise a FileNotFoundError.

        :param data: List of image to process
        :param classifier: Classifier implementation
        :return: Probability that the data is a web shop
        """
        values = self.__generate_values(data, classifier)
        max_value = 0
        # Search max index in probabilities (number of categories)
        for val in values:
            if val[0] > max_value:
                max_value = val[0]
        # Add element to the array (start at 0)
        max_value += 1
        # List of probabilities for each category
        probabilities = [0] * max_value
        for val in values:
            if val[1] > probabilities[val[0]]:
                probabilities[val[0]] = val[1]
        data = 0
        for prob in probabilities:
            if (prob*100) > data:
                data = (prob*100)
        return data

    def train_classifier(self, data, targets, classifier):
        """
        Train the classifier with the feature for later use.

        :param data: Images for the training process (In this case can be also None)
        :param targets: Images targets for training process
        :param classifier: Classifier implementation
        :return: Nothing
        """
        # Load default parameters
        parameters = self.__load_default()
        # Load the histograms generated in the training phase
        old_histogram = np.loadtxt(
            os.path.join(self.feature_directory, 'default_' + '_'.join(parameters), "histogram.out"))
        # HARD NEGATIVE UPDATE
        # Load bounding boxes file
        lines = [line.rstrip('\n') for line in open(self.bounding_box_file)]
        # Load vocabulary generated in the training process
        self.vocabulary = np.loadtxt(os.path.join(self.feature_directory, 'default_' + '_'.join(parameters), 'voc.out'))
        # Load standard scaler generated in the training process
        self.std_slr = pickle.load(
            open(os.path.join(self.feature_directory, 'default_' + '_'.join(parameters), "std.out"), "rb"))
        # Loop over all categories
        for i in range(len(targets)):
            # Classifier train with class i
            classifier.train(old_histogram, targets[i], '_'.join(parameters) + '_old')
            # Histograms for fit
            histograms = []
            # Classes for fit
            classes = []
            #  Search for the start of data for the i-th class
            start = lines.index(str(i)) + 1
            # If it is last class end is last element
            if i == len(targets) - 1:
                end = len(lines) - 1
            # If it is not the last class go until the element before the next class
            else:
                end = lines.index(str(i + 1), start)
            # Loop over the different images in the banner_dir
            for k, dir_name in enumerate(os.listdir(self.banner_dir)):
                boxes = []
                # Load all data for class i
                for j in range(start, end):
                    # Load only boxes for current image
                    if int(lines[j].split("_")[0]) == k:
                        # Loading bounding box
                        boxes.append(list(map(int, lines[j].split("_")[1].split(","))))
                # Add border to the image (cv2 bug)
                bordersize = 200
                border = cv2.copyMakeBorder(cv2.imread(os.path.join(self.banner_dir, dir_name)), top=bordersize,
                                            bottom=bordersize, left=bordersize, right=bordersize,
                                            borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                # Set input image for segmentation
                ss.setBaseImage(border)
                ss.switchToSelectiveSearchQuality()
                # run selective search segmentation on input image
                rects = ss.process()
                # iterate over all the region proposals
                for rect_num, rect in enumerate(rects):
                    # loop only over 200 bounding boxes
                    if rect_num < 200:
                        x, y, w, h = rect
                        # IoU check
                        check = False
                        # Iterate over loaded boxes
                        for box in boxes:
                            # Check if there is a intersection over union
                            if self.__bb_intersection_over_union(box, [x, y, x + w, y + h]) > 0:
                                check = True
                                break
                        # Compute bag of word histogram
                        hist = self.__compute_hist(border[y:y + h, x:x + w])
                        if hist is not None:
                            # Train the classifier that it said no
                            if not check and classifier.classify([hist], '_'.join(parameters) + '_old')[0][0] >= 0.5:
                                classes.append(1)
                                histograms.append(hist[0])
                    else:
                        break
            # Train the classifier with the hard negative update
            if len(classes) != 0:
                # Fit classifier with new data
                classifier.train(np.append(np.array(histograms), old_histogram, axis=0),
                                 np.append(np.array(classes), targets[i], axis=0), '_'.join(parameters) + '_' + str(i))
            # If the hard negative update did not give any results, train the system normally
            else:
                classifier.train(old_histogram, targets[i], '_'.join(parameters) + '_' + str(i))

    def need_train(self):
        """
        Get if the feature need a train.

        :return: True if it need a train otherwise False
        """
        # This feature need a train phase
        return True

    def train(self, data):
        """
        Train the feature with the given data. Bow need a K-Mean clustering which is very long. It must generate a
        vocabulary of visual words.

        :param data: List of images
        :return: Nothing
        """
        # Construct SIFT object
        sift = cv2.xfeatures2d.SIFT_create()
        # SIFT descriptors list
        descriptors = []
        # Loops over all images and compute SIFT descriptors
        for img_path in data:
            gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            # Compute SIFT
            kp, des = sift.detectAndCompute(gray, None)
            # Append image SIFT descriptors to main array
            descriptors.append(des)
        # descriptors len = number of images
        # descriptors[x] len = number of feature for images
        # descriptors[x][y] len = 128 -> SIFT descriptor

        # Put data together
        t_descriptors = descriptors[0][1]
        for i, descriptor in enumerate(descriptors[1:]):
            if isinstance(descriptor, np.ndarray):
                t_descriptors = np.vstack((t_descriptors, descriptor))

        # t_descriptor len = total number of images features
        # t_descriptor[x] len = 128 -> SIFT descriptor

        parameters = self.__load_default()

        # VOCABULARY GENERATION
        # Stop the iteration when any of the condition is met (accuracy and max number of iterations)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(parameters[1]), 1.0)
        # K-mean clustering, vocabulary has k words
        ret, label, vocabulary = cv2.kmeans(np.float32(t_descriptors), int(parameters[0]), None, criteria,
                                            int(parameters[2]),
                                            cv2.KMEANS_RANDOM_CENTERS)

        # Save vocabulary
        np.savetxt(os.path.join(self.feature_directory, 'default_' + '_'.join(parameters), 'voc.out'), vocabulary)

        # Compute histograms
        # Histograms generation array of (images count, number of words)
        histograms = np.zeros((len(data), int(parameters[0])))
        for i in range(len(data)):
            if isinstance(descriptors[i], np.ndarray):
                # Assign codes from a code book to observations.
                words, distance = vq(descriptors[i], vocabulary)
                for w in words:
                    histograms[i][w] += 1

        # Create standard scaler and standardize data
        std_slr = sk.StandardScaler().fit(histograms)
        # Save standard scaler
        pickle.dump(std_slr,
                    open(os.path.join(self.feature_directory, 'default_' + '_'.join(parameters), "std.out"), "wb"))

        if parameters[3] == 'True':
            histograms = std_slr.transform(histograms)

        # Save histograms and standard scaler
        np.savetxt(os.path.join(self.feature_directory, 'default_' + '_'.join(parameters), "histogram.out"), histograms)

    def show(self):
        """
        Show all the available variations.

        :return: List of all variations (Lists of parameters)
        """
        names = [name for name in os.listdir(self.feature_directory)]
        params = []
        for n in names:
            params.append(' '.join(n.split('_')))
        return params

    def delete(self, parameters):
        """
        Delete the given variation. After the removal, a new default variation must be set. Raise  aFileNotFoundError if
        the given parameters do not represent an existent variation.

        :param parameters: K (K-Means), Iteration (K-Means), Attempts (K-Means), Standardization (True or False)
        :return: Nothing
        """
        # Check if the variation exists
        if not self.__check_variation_existence(parameters):
            raise FileNotFoundError
        # Loop over all directories (variations)
        names = [name for name in os.listdir(self.feature_directory)]
        for n in names:
            check = True
            # If default it must take only the last 4 parameters to compare
            if 'default' in n:
                params = n.split('_')
                for a, b in zip(parameters, params[1:]):
                    if a != b:
                        check = False
                        break
                if check:
                    shutil.rmtree(os.path.join(self.feature_directory, n))
                    break
            # All others
            else:
                for a, b in zip(parameters, n.split('_')):
                    if a != b:
                        check = False
                        break
                if check:
                    shutil.rmtree(os.path.join(self.feature_directory, n))
                    break

    def add(self, parameters):
        """
        Add a new variation to the system. The new variation is set as default automatically. If the variation already
        exist, raise a FileExistsError. If the parameters are not of the right format raise a AttributeError.

        :param parameters: K (K-Means), Iteration (K-Means), Attempts (K-Means), Standardization (True or False)
        :return: Nothing
        """
        if len(parameters) != 4 or not self.__check_int(parameters[0]) or not self.__check_int(
                parameters[1]) or not self.__check_int(parameters[2]) or not self.__check_bool(parameters[3]):
            raise AttributeError

        if self.__check_variation_existence(parameters):
            raise FileExistsError
        else:
            name = '_'.join(parameters)
            os.makedirs(os.path.join(self.feature_directory, name))
            self.set_default(parameters)

    def set_default(self, parameters):
        """
        Set as default the given feature variation. Raise a FileNotFoundError if the given parameters do not represent
        an existent variation.

        :param parameters: K (K-Means), Iteration (K-Means), Attempts (K-Means), Standardization (True or False)
        :return: Nothing
        """
        # Check if the variation exists
        if not self.__check_variation_existence(parameters):
            raise FileNotFoundError
        # Loop over all folder
        names = [name for name in os.listdir(self.feature_directory)]
        for n in names:
            # Remove default
            if 'default' in n:
                d = False
                params = n.split('_')
                # Only if it is not the same as the given parameters
                for a, b in zip(parameters, params[1:]):
                    if a != b:
                        os.rename(os.path.join(self.feature_directory, n),
                                  os.path.join(self.feature_directory, '_'.join(params[1:])))
                        d = True
                        break
                if d:
                    continue
            # Set to default the new variation
            else:
                d = False
                for a, b in zip(parameters, n.split('_')):
                    if a != b:
                        d = True
                        break
                if d:
                    continue
                os.rename(os.path.join(self.feature_directory, n), os.path.join(self.feature_directory, 'default_' + n))

    def is_trained(self):
        """
        Get if the default Bow variation is trained or not.

        :return: True if it is trained or it does not need a training phase, otherwise False
        """
        # Load all feature variations
        names = [name for name in os.listdir(self.feature_directory)]
        # Loop over all directories
        for n in names:
            # If the default variation has file in it -> trained!
            if 'default' in n and os.listdir(os.path.join(self.feature_directory, n)) != []:
                return True
        return False

    def default_exist(self):
        """
        Get if the a default variation exists.

        :return: True if the default variation exists otherwise False
        """
        names = [name for name in os.listdir(self.feature_directory)]
        # Loop over all directories
        for n in names:
            # If exists a default implementation
            if 'default' in n:
                return True
        return False

    # ############################ HELPER ############################

    @staticmethod
    def __check_bool(s):
        """
        Check that the given string is a boolean.

        :param s: Input string
        :return: True if the input is a boolean otherwise False
        """
        if s == 'True' or s == 'False':
            return True
        else:
            return False

    @staticmethod
    def __check_int(s):
        """
        Check that the given string is a integer.

        :param s: Input string
        :return: True if the input is an integer otherwise False
        """
        try:
            int(s)
            return True
        except ValueError:
            return False

    def __load_default(self):
        """
        Get parameters of the default implementation.

        :return: Parameters of the default implementation
        """
        names = [name for name in os.listdir(self.feature_directory)]
        for n in names:
            if 'default' in n:
                l = n.split('_')
                return l[1:]
        return None

    def __check_variation_existence(self, parameters):
        """
        Check if the given variation exists.

        :param parameters: K (K-Means), Iteration (K-Means), Attempts (K-Means), Standardization (True or False)
        :return: True if the variation exist otherwise False
        """
        name = '_'.join(parameters)
        if os.path.exists(os.path.join(self.feature_directory, name)) or os.path.exists(
                os.path.join(self.feature_directory, 'default_' + name)):
            return True
        else:
            return False

    @staticmethod
    def __bb_intersection_over_union(box_a, box_b):
        """
        Compute intersection over union
        see https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/ .

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
        # areas - the intersection area
        try:
            iou = interArea / float(boxAArea + boxBArea - interArea)
            # return the intersection over union value
            return iou
        except ZeroDivisionError:
            return 0

    def __compute_hist(self, img):
        """
        Compute histogram with sift descriptors and vocabulary. The vocabulary and standard scaler must be loaded before
        ca this method (self.vocabulary and self.std_slr)

        :param img: image to compute histogram
        :return: generated histogram
        """
        # Load default parameters
        parameters = self.__load_default()
        # Construct SIFT object
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute SIFT
        kp, descriptors = sift.detectAndCompute(gray, None)
        # if descriptor are > 0 and < 200 compute histogram
        if descriptors is not None and len(descriptors) < 200:
            histograms = np.zeros((1, int(parameters[0])))
            # Assign codes from a code book to observations.
            words, distance = vq(descriptors, self.vocabulary)
            for w in words:
                histograms[0][w] += 1
            # Standardize
            histograms = self.std_slr.transform(histograms)
        else:
            histograms = None
        return histograms

    def __generate_values(self, data, classifier):
        """
        Generate probability for each category and each image of the data set.

        :param data: List of images
        :param classifier: Classifier implementation
        :return: List of probability and category pair (category,prob)
        """
        # Load default parameters
        parameters = self.__load_default()

        # Load classifiers
        classifiers = []
        z = 0
        while True:
            try:
                classifier.classify([], '_'.join(parameters) + '_' + str(z))
                z += 1
            except FileNotFoundError:
                break

        # The classifier is not trained
        if z < 2:
            raise FileNotFoundError

        # OpenCV Selective Search has a bug which generates bounding boxes outside the border of the image.
        # Therefore, a big border is needed to solve the problem.
        bordersize = 200
        # List for all the Histograms
        all = []
        # Load vocabulary generated in the training process
        self.vocabulary = np.loadtxt(os.path.join(self.feature_directory, 'default_' + '_'.join(parameters), 'voc.out'))
        # Load standard scaler generated in the training process
        self.std_slr = pickle.load(
            open(os.path.join(self.feature_directory, 'default_' + '_'.join(parameters), "std.out"), "rb"))
        # Loop over all images
        for img in data:
            probabilities = [0] * z
            # Apply white border
            border = cv2.copyMakeBorder(cv2.imread(img), top=bordersize,
                                        bottom=bordersize, left=bordersize, right=bordersize,
                                        borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
            # Create Selective Search Segmentation Object using default parameters
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            # Set input image on which we will run segmentation
            ss.setBaseImage(border)
            # Use quality Search
            ss.switchToSelectiveSearchQuality()
            # Run selective search segmentation on input image
            rects = ss.process()
            # Iterate over all the region proposals
            for j, rect in enumerate(rects):
                # Compare rectangle for region proposal till numShowRects
                if j < 500:  # 500 is the best value found in the Bachelor Thesis Logos recognition for website services
                    # Get rectangle coordinates
                    x, y, w, h = rect
                    probs = []
                    # Loop over the different logos classes to compute the histogram
                    for i in range(z):
                        hist = self.__compute_hist(border[y:y + h, x:x + w])
                        if hist is not None:
                            # Compute probability that the histogram is the logo i
                            prob = classifier.classify([hist], '_'.join(parameters) + '_' + str(i))[0]
                            # Save probability and logo type
                            probs.append((prob[0], i))
                    # Sort the probabilities to have the highest as the first element
                    probs.sort(key=lambda tup: tup[0], reverse=True)
                    # loop over all probabilities (one for each class) and take the highest until now
                    for i in range(len(probs)):
                        # if prob is higher than the probability until now replace it
                        if probs[i][0] > probabilities[probs[i][1]]:
                            probabilities[probs[i][1]] = probs[i][0]
                            # print image with random name
                            break
                else:
                    break
            for i in range(z):
                all.append((i, probabilities[i]))
        return all
