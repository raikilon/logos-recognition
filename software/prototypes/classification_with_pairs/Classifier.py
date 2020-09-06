from Features import Features
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import logging
import os
import sys
import datetime


class Classifier(object):
    def __init__(self):

        tests = [("svm", "linear"), ("svm", "poly"), ("svm", "rbf"), ("svm", "sigmoid"), ("svm", "svc"),
                 ("bayes", "gaussian"), ("bayes", "bernoulli"), ("knn", 11), ("knn", 53),
                 ("knn", 101), ("dtree", "gini"), ("dtree", "entropy"), ("rforest", 10), ("rforest", 100),
                 ("rforest", 1000), ("rforest", 10000)
                 ]
        # ("bayes", "multinomial"), cannot have negative values
        # To use it train the data with data_standardization = False

        # ----------------- SETTINGS -----------------
        generate_data = False
        data_standardization = True
        k_values = [10000]
        iterations = [10, 20, 50]
        attempts = [10, 20, 50]
        max_voc_size = [10000, 20000, 50000, 100000]
        train_dir = os.path.abspath("C:/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/train")
        test_dir = os.path.abspath("C:/git/Logos-Recognition-for-Webshop-Services/logorec/resources/images/test")


        # Starting logging
        logging.basicConfig(filename='data.log', filemode='w', level=logging.INFO)

        if generate_data:
            feature = Features("data", train_dir, os.listdir(train_dir)[0])
            logging.warning("Generating data for train and test ...")
            # Generate data for each logo types (e.g. MasterCard vs Other, Visa vs Other, etc.)
            for k in k_values:
                for a in attempts:
                    for i in iterations:
                        for s in max_voc_size:
                            logging.info("Number of cluster (kmeans): " + str(k))
                            logging.info("Voc Size: " + str(s))
                            logging.info("Number of iterations: " + str(i))
                            logging.info("Number of attempts: " + str(a))
                            start_time = datetime.datetime.now()
                            feature.train(k, data_standardization, False, s, i, a)
                            logging.info("Time train: " + str(datetime.datetime.now() - start_time))
            logging.warning("Generation ended.")

        logging.warning("Start test ...")
        for algo in tests:
            for k in k_values:
                for a in attempts:
                    for it in iterations:
                        for s in max_voc_size:
                            # Confusion matrix quality values initialisation
                            tpr = 0
                            tnr = 0
                            ppv = 0
                            npv = 0
                            acc = 0
                            # Number of total classes (Visa vs Other, MasterCard vs Other, etc.)
                            classes = 0
                            # Time initialisation
                            start_time = datetime.datetime.now()
                            # Loop for each logo types (e.g. MasterCard vs Other, Visa vs Other, etc.)
                            for dir_name in os.listdir(train_dir):

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
                                # lead train data
                                feature = Features("data", train_dir, dir_name)
                                classifier.fit(feature.get_train_histograms(k), feature.images_class)

                                # Prediction generation
                                # ----------------- TO PRINT CLASSES NUMBERS USE THIS TWO LINES -----------------
                                feature = Features("data", test_dir, dir_name)
                                solutions = classifier.predict(feature.test(k, data_standardization, s, it, a))

                                # Confusion matrix values initialisation
                                tp = 0
                                tn = 0
                                fn = 0
                                fp = 0
                                # Confusion matrix values computation
                                for i in range(len(solutions)):
                                    if int(solutions[i]) == int(feature.images_class[i]):
                                        if feature.images_class[i] == 1:
                                            tn += 1
                                        else:
                                            tp += 1
                                    else:
                                        if feature.images_class[i] == 1:
                                            fp += 1
                                        else:
                                            fn += 1
                                # Confusion matrix values computation

                                try:
                                    # temp values because can happen division by zero
                                    t_tpr = tp / (tp + fn)
                                    t_tnr = tn / (tn + fp)
                                    t_ppv = tp / (tp + fp)
                                    t_npv = tn / (tn + fn)
                                    t_acc = (tp + tn) / (tp + tn + fp + fn)
                                except ZeroDivisionError:
                                    exc_type, exc_obj, exc_tb = sys.exc_info()
                                    logging.error(
                                        "Division by zero for " + str(algo[0]) + ": " + str(
                                            algo[1]) + " with: " + dir_name + " at line: " + str(exc_tb.tb_lineno))

                                else:
                                    tpr += t_tpr
                                    tnr += t_tnr
                                    ppv += t_ppv
                                    npv += t_npv
                                    acc += t_acc
                                    classes += 1

                            # Print information
                            try:
                                logging.info("Algorithm: " + str(algo[0]) + ": " + str(algo[1]))
                                logging.info("Number of cluster (kmeans): " + str(k))
                                logging.info("Voc size: " + str(s))
                                logging.info("Number of iterations: " + str(it))
                                logging.info("Number of attempts: " + str(a))
                                logging.info("Time: " + str(datetime.datetime.now() - start_time))
                                logging.info("Sensitivity: %.1f" % ((tpr / classes) * 100))
                                logging.info("Specificty: %.1f" % ((tnr / classes) * 100))
                                logging.info("Precision: %.1f" % ((ppv / classes) * 100))
                                logging.info("Negative Predictive Value: %.1f" % ((npv / classes) * 100))
                                logging.info("Accuracy: %.1f" % ((acc / classes) * 100))
                                logging.info("_______________________________")
                            except ZeroDivisionError:
                                logging.error("Error")
                                logging.info("_______________________________")


def main():
    classifier = Classifier()


if __name__ == "__main__":
    main()
