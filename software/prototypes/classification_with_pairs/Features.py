import os
import cv2
import numpy as np
import sklearn.preprocessing as sk
from scipy.cluster.vq import *
from matplotlib import pyplot as plt
import pickle


class Features(object):
    def __init__(self, data_directory, images, current_class):
        """ Initialise variables

        :param data_directory: directory to store all the generated data
        :param images: images to test (train or test)
        :param current_class: the current tested class (DHL, MasterCard, Etc.)
        """
        # Path to all images
        images_path = []
        # List of images classes
        images_class = []
        # Generate data directory if it does not exist
        os.makedirs(data_directory, exist_ok=True)
        # Loop over all folder
        for train_images_subdir in os.listdir(images):
            # Train images subdirectory
            subdir = os.path.join(images, train_images_subdir)
            # All images in the logo directory
            class_path = [os.path.join(subdir, f) for f in os.listdir(subdir)]
            # Add images path to main array
            images_path += class_path
            if train_images_subdir != current_class:
                # Class 1 is for the logos that are not of the same type of the tested logo
                # Add images class to main array (len(class_paths) times 1)
                images_class += [1] * len(class_path)
            elif train_images_subdir == current_class:
                # Class 0 is for the tested logo
                images_class += [0] * len(class_path)
        # Save the classes array
        np.savetxt(data_directory + "/classes_" + current_class + ".out", images_class)
        # Save data for later use
        self.data_directory = data_directory
        self.images_path = images_path
        self.images_class = images_class

    def train(self, k=100, standardization=True, draw_hist=False, max_voc_size=10000, iterations=10, attempts=10):
        """ Generate descriptors, vocabulary and histograms and save them for later uses

        :param k: number of cluster for k-means
        :param standardization: True for standardisation otherwise false (Default: True)
        :param draw_hist: Draw histogram (Default:False)
        :param max_voc_size: max number of descriptors for the vocabulary
        :param iterations: max number of iteration fo the kmeans to obtain the best values
        :param attempts: max number of restart that the kmeans can perform
        :return: nothing
        """
        # computer descriptors
        descriptors = self.compute_descriptors()

        # Put data together
        t_descriptors = descriptors[0][1]
        for i, descriptor in enumerate(descriptors[1:]):
            if isinstance(descriptor, np.ndarray):
                if i > max_voc_size:
                    break
                t_descriptors = np.vstack((t_descriptors, descriptor))

        # t_descriptor len = total number of images features
        # t_descriptor[x] len = 128 -> SIFT descriptor

        # VOCABULARY GENERATION
        # Stop the iteration when any of the condition is met (accuracy and max number of iterations)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
        # K-mean clustering, vocabulary has k words
        ret, label, vocabulary = cv2.kmeans(np.float32(t_descriptors), k, None, criteria, attempts,
                                            cv2.KMEANS_RANDOM_CENTERS)

        # Save vocabulary
        np.savetxt(
            self.data_directory + "/voc_" + str(k) + "_" + str(max_voc_size) + "_" + str(iterations) + "_" + str(attempts) + ".out",
            vocabulary)

        # Compute histograms
        histograms = self.compute_histograms(descriptors, vocabulary, k)

        # Create standard scaler and standardize data
        std_slr = sk.StandardScaler().fit(histograms)
        if standardization:
            histograms = std_slr.transform(histograms)

        # Save histograms and standard scaler
        np.savetxt(self.data_directory + "/histograms_" + str(k) + ".out", histograms)
        pickle.dump(std_slr, open(self.data_directory + "/std_slr_" + str(k) + ".out", "wb"))

        # draw histograms
        if draw_hist:
            for i in range(len(histograms)):
                self.draw_histogram(histograms, k, i)

    def get_train_histograms(self, k):
        """
        Load histogram from data directory
        :param k: the number of cluster
        :return:
        """
        return np.loadtxt(self.data_directory + "/histograms_" + str(k) + ".out")

    def test(self, k=100, standardization=True, max_voc_size=10000, iterations=10, attempts=10):
        """ Computer descriptor and histogram using the previously generated vocabulary

        :param standardization: True for standardisation otherwise false (Default: True)
        :param k: number of cluster for k-means
        :param max_voc_size: max number of descriptors for the vocabulary
        :param iterations: max number of iteration fo the kmeans to obtain the best values
        :param attempts: max number of restart that the kmeans can perform
        :return: generated histogram
        """
        descriptors = self.compute_descriptors()
        vocabulary = np.loadtxt(
            self.data_directory + "/voc_" + str(k) + "_" + str(max_voc_size) + "_" + str(iterations) + "_" + str(attempts) + ".out")
        histograms = self.compute_histograms(descriptors, vocabulary, k)
        if standardization:
            std_slr = pickle.load(open(self.data_directory + "/std_slr_" + str(k) + ".out", "rb"))
            histograms = std_slr.transform(histograms)
        return histograms

    def compute_histograms(self, descriptors, vocabulary, k):
        """ Create histograms based on the descriptor and the number of cluster

        :param descriptors: SIFT descriptor to generate the histograms
        :param vocabulary: vocabulary previously generated with kmeans
        :param k: number of cluster of kmeans and therefore number of elements of the histogram
        :return:
        """
        # Histograms generation array of (images count, number of words)
        histograms = np.zeros((len(self.images_path), k))
        for i in range(len(self.images_path)):
            if isinstance(descriptors[i], np.ndarray):
                # Assign codes from a code book to observations.
                words, distance = vq(descriptors[i], vocabulary)
                for w in words:
                    histograms[i][w] += 1
        return histograms

    def compute_descriptors(self):
        """ Generate SIFT descriptor from images

        :return: descriptors
        """
        # Construct SIFT object
        sift = cv2.xfeatures2d.SIFT_create()
        # SIFT descriptors list
        descriptors = []
        # Loops over all images and compute SIFT descriptors
        for img_path in self.images_path:
            gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            # Compute SIFT
            kp, des = sift.detectAndCompute(gray, None)
            # Append image SIFT descriptors to main array
            descriptors.append(des)
        # descriptors len = number of images
        # descriptors[x] len = number of feature for images
        # descriptors[x][y] len = 128 -> SIFT descriptor
        return descriptors

    def draw_histogram(self, histograms, k, i):
        """
        Draw an histogram from the i element of the generated histograms with k values
        :param histograms: histograms previously generated
        :param k: number of cluster used in kmeans
        :param i: index of the histogram to print
        :return: nothing
        """
        x_scalar = np.arange(k)
        y_scalar = np.array([abs(np.sum(histograms[i, h], dtype=np.int32)) for h in range(k)])

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()


def main():
    # ----------------- HOW TWO USE -----------------
    # 1) create a Features class
    # 2) call train() to generate data (new vocabulary)
    # 3) call test() to generate test data with previous generated vocabulary
    features = Features()


if __name__ == "__main__":
    main()
