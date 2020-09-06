from Features import Features
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import logging
import os
import sys
import datetime


class Classifier(object):
    """
       ----------------- OUTDATED -----------------
       This prototype is not more up to date with the other. It has been abandoned because its results were to low
    """
    def __init__(self):
        # ----------------- SETTINGS -----------------
        generate_data = True
        data_standardization = True

        # ----------------- ALGORITHMS -----------------
        tests = [("svm", "linear"), ("svm", "poly"), ("svm", "rbf"), ("svm", "sigmoid"), ("svm", "svc"),
                 ("bayes", "gaussian"), ("bayes", "bernoulli"), ("knn", 11), ("knn", 53),
                 ("knn", 101), ("dtree", "gini"), ("dtree", "entropy"), ("rforest", 10), ("rforest", 100),
                 ("rforest", 1000), ("rforest", 10000)]

        # ("bayes", "multinomial") cannot have negative values
        # To use it train the data with data_standardization = False

        k_values = [100, 500, 1000, 10000]

        # Images directories
        # ----------------- CHANGE TRAIN AND TEST DIRECTORY -----------------
        train_dir = os.path.abspath("C:/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/train")
        test_dir = os.path.abspath("C:/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/test")

        # Start logging
        logging.basicConfig(filename='data.log', filemode='w', level=logging.INFO)

        if generate_data:
            logging.warning("Generating data for train and test ...")
            # Generate data for each logo types (e.g. MasterCard vs Other, Visa vs Other, etc.)
            for k in k_values:
                logging.info("Number of cluster (kmeans): " + str(k))
                start_time = datetime.datetime.now()
                feature = Features()
                feature.generate_data(train_dir, k, data_standardization)
                feature.save_data("train/" + str(k))
                logging.info("Time train: " + str(datetime.datetime.now() - start_time))
                start_time = datetime.datetime.now()
                t_feature = Features()
                t_feature.generate_data(test_dir, k, data_standardization, False,
                                        feature.vocabulary, feature.std_slr)
                t_feature.save_data("test/" + str(k))
                logging.info("Time test: " + str(datetime.datetime.now() - start_time))
            logging.warning("Generation ended.")

        feature = Features()
        logging.warning("Start test ...")
        for algo in tests:
            for k in k_values:
                # Time initialisation
                start_time = datetime.datetime.now()
                # lead train data
                feature.load_data("train/" + str(k))
                # CLASSIFIER
                if algo[0] == "svm":
                    # SVC
                    if algo[1] == "svc":
                        classifier = svm.LinearSVC()
                    else:
                        classifier = svm.SVC(kernel=algo[1])
                if algo[0] == "bayes":
                    # Bayes
                    if algo[1] == "multinomial":
                        classifier = MultinomialNB()
                    elif algo[1] == "gaussian":
                        classifier = GaussianNB()
                    else:
                        classifier = BernoulliNB()
                if algo[0] == "dtree":
                    # Decision tree
                    if algo[1] == "gini":
                        classifier = tree.DecisionTreeClassifier()
                    else:
                        classifier = tree.DecisionTreeClassifier(criterion="entropy")
                if algo[0] == "knn":
                    # knn
                    classifier = KNeighborsClassifier(algo[1])
                if algo[0] == "rforest":
                    # Random Forest
                    classifier = RandomForestClassifier(algo[1])

                classifier.fit(feature.histograms, feature.main_images_class)

                feature.load_data("test/" + str(k))

                # Prediction generation
                solutions = classifier.predict(feature.histograms)

                confusion = confusion_matrix(feature.main_images_class, solutions)

                # Print information
                logging.info("Algorithm: " + str(algo[0]) + ": " + str(algo[1]))
                logging.info("Number of cluster (kmeans): " + str(k))
                logging.info("Time: " + str(datetime.datetime.now() - start_time))
                logging.info("Confusion: ")
                logging.info(confusion)


def main():
    classifier = Classifier()


if __name__ == "__main__":
    main()
