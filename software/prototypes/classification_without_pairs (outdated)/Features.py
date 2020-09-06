import os
import cv2
import numpy as np
import sklearn.preprocessing as sk
from scipy.cluster.vq import *
from matplotlib import pyplot as plt
import datetime
import pickle


class Features(object):
    """
    ----------------- OUTDATED -----------------
    This prototype is not more up to date with the other. It has been abandoned because its results were to low
    """
    def __init__(self):
        """
        Instantiate initial parameters
        """
        # Generated histograms
        self.histograms = []
        # Vocabulary
        self.vocabulary = []
        # Standard scaler
        self.std_slr = None
        # Path to all images
        self.main_images_path = []
        # List of images classes
        self.main_images_class = []

    def extract_directories_information(self, images):
        """
        Extract information about the images in the given folder. It saves the images path and images classes.

        :param images: absolute path to processed images
        :return: nothing
        """

        class_id = 0
        # Loops over train images subdirectories
        for imgs_subdir in os.listdir(images):
            # Train images subdirectory
            subdir = os.path.join(images,imgs_subdir)
            # Images paths
            class_path = [os.path.join(subdir, f) for f in os.listdir(subdir)]
            # Images path - This array contains the path of each image
            self.main_images_path += class_path
            # Images class - This array contains the class of each images path
            self.main_images_class += [class_id] * len(class_path)
            # Increase class ID - Each folder is a new class
            class_id += 1

    def draw_histogram(self, k, i):
        """
        Draw an histogram from the i element of the generated histograms with k values
        :param k:
        :param i:
        :return: nothing
        """
        x_scalar = np.arange(k)
        y_scalar = np.array([abs(np.sum(self.histograms[i, h], dtype=np.int32)) for h in range(k)])

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()

    def generate_data(self, images, k=100, standardization=True, train=True, vocabulary=[], std_slr=None,
                      draw_hist=False):
        """
        Computes SIFT feature, performs k-means and compute normalized histograms

        :param images: absolute path to processed images
        :param k: number of cluster
        :param train: if train phase True otherwise False (default: True)
        :param vocabulary: vocabulary for test phase (default: empty)
        :param std_slr: standard scaler (default: None)
        :param draw_hist: True to draw histograms otherwise False
        :return: nothing
        """

        # Extract information
        self.extract_directories_information(images)

        # ***** SIFT DESCRIPTORS GENERATION *****

        # Construct SIFT object
        sift = cv2.xfeatures2d.SIFT_create()

        # SIFT descriptors list
        descriptors = []
        # Loops over all images and compute SIFT descriptors
        for img_path in self.main_images_path:
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
        for descriptor in descriptors[1:]:
            t_descriptors = np.vstack((t_descriptors, descriptor))

        # t_descriptor len = total number of images features
        # t_descriptor[x] len = 128 -> SIFT descriptor

        # ***** VOCABULARY GENERATION *****

        if train:
            # Stop the iteration when any of the condition is met (accuracy and max number of iterations)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # K-mean clustering, vocabulary has k words
            ret, label, self.vocabulary = cv2.kmeans(np.float32(t_descriptors), k, None, criteria, 10,
                                                     cv2.KMEANS_RANDOM_CENTERS)
        else:
            self.vocabulary = vocabulary

        # ***** HISTOGRAM GENERATION *****

        # Histograms generation array of (images count, number of words)
        histograms = np.zeros((len(self.main_images_path), k))
        for i in range(len(self.main_images_path)):
            # Assign codes from a code book to observations.
            words, distance = vq(descriptors[i], self.vocabulary)
            for w in words:
                histograms[i][w] += 1

        # Standardize histograms by removing the mean and scaling to unit variance
        if train:
            # If train phase compute the mean and std to be used for later scaling
            self.std_slr = sk.StandardScaler().fit(histograms)
        else:
            self.std_slr = std_slr

        # Perform standardization by centering and scaling
        if standardization:
            self.histograms = self.std_slr.transform(histograms)
        else:
            self.histograms = histograms

        if draw_hist:
            for i in range(len(histograms)):
                self.draw_histogram(k, i)

    def save_data(self, data_path="train"):
        """
        Saves the important information to the given directory. If the directory does not exist it create a new one

        :param data_path: relative path to where save the data (default: train)
        :return: nothing
        """
        # Create folder if it does not exist
        os.makedirs(data_path, exist_ok=True)
        # Save important data
        np.savetxt(data_path + "/data.out", self.histograms,
                   footer="Generated on " + datetime.date.today().strftime("%d.%m.%Y"))
        np.savetxt(data_path + "/classes.out", self.main_images_class,
                   footer="Generated on " + datetime.date.today().strftime("%d.%m.%Y"))
        np.savetxt(data_path + "/classes_name.out", self.main_images_path,
                   footer="Generated on " + datetime.date.today().strftime("%d.%m.%Y"), fmt="%s")
        np.savetxt(data_path + "/voc.out", self.vocabulary,
                   footer="Generated on " + datetime.date.today().strftime("%d.%m.%Y"))
        pickle.dump(self.std_slr, open(data_path + "/std_slr.out", "wb"))

    def load_data(self, data_path="train"):
        """
        Loads data if the files exist in the data_path folder. If they do not exist they are computed from scratch
        :param data_path: relative path to where data is saved (default: train)
        :return: nothing
        """
        if os.path.isdir(data_path) and os.path.exists(data_path + "/data.out") and os.path.exists(
                data_path + "/classes.out") and os.path.exists(data_path + "/voc.out") and os.path.exists(
            data_path + "/std_slr.out"):
            self.histograms = np.loadtxt(data_path + "/data.out")
            self.main_images_class = np.loadtxt(data_path + "/classes.out")
            self.vocabulary = np.loadtxt(data_path + "/voc.out")
            self.std_slr = pickle.load(open(data_path + "/std_slr.out", "rb"))
        else:
            self.generate_data(data_path)
            self.save_data()


def main():
    features = Features("C:/Dev/Images/")
    features.generate_data()
    features.save_data()


if __name__ == "__main__":
    main()
