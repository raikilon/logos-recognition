from lib.classifier import Classifier
import os
import pickle
import shutil
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Classifier):
    """
    A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the
    data set and use averaging to improve the predictive accuracy and control over-fitting.
    """

    def __init__(self):
        """
        Initiate the classifier by creating the Random Forest directory in data/classifier.
        """
        # data directory
        self.classifier_directory = 'data/classifier/randomForest'
        # Generate classifier directory if it does not exist
        if not os.path.exists(self.classifier_directory):
            os.makedirs(self.classifier_directory)

    def classify(self, samples, name):
        """
        Load the given classifier variation (and use it to classify the samples. If the classifier implementation does
        not exist, raise a FileNotFoundError.

        :param samples: Samples generated with a feature
        :param name: Name of the classifier variation
        :return: List of probabilities
        """
        parameters = self.__load_default()
        if os.path.isfile(self.__get_default_name(name)):
            classifier = pickle.load(
                open(self.__get_default_name(name), "rb"))
            probabilities = []
            for sample in samples:
                prob = classifier.predict_proba(sample)[0]
                probabilities.append(prob)
            return probabilities
        else:
            raise FileNotFoundError

    def train(self, samples, targets, name):
        """
        Train the classifier with the given samples and targets and then save the results with the given name.

        :param samples: Samples generated with a feature
        :param targets: Targets pf the samples for the training process
        :param name: Classifier variation name
        :return: Nothing
        """
        parameters = self.__load_default()
        classifier = RandomForestClassifier(int(parameters[0]))
        classifier.fit(samples, targets)
        pickle.dump(classifier,
                    open(self.__get_default_name(name),
                         "wb"))

    def is_trained(self):
        """
        Get if the default random forest variation is trained or not.

        :return: True if it is trained, otherwise False
        """
        # Load all classifier variations
        names = [name for name in os.listdir(self.classifier_directory)]
        # Loop over all directories
        for name in names:
            # If the default variation has file in it -> trained!
            if 'default' in name and os.listdir(os.path.join(self.classifier_directory, name)) != []:
                return True
        return False

    def default_exist(self):
        """
        Get if the a default variation exists.

        :return: True if the default variation exists otherwise False
        """
        names = [name for name in os.listdir(self.classifier_directory)]
        # Loop over all directories
        for name in names:
            # If exists a default implementation
            if 'default' in name:
                return True
        return False

    def show(self):
        """
        Show all the available variations.

        :return: List of all variations (List of list of parameters)
        """
        names = [name for name in os.listdir(self.classifier_directory)]
        params = []
        for n in names:
            params.append(' '.join(n.split('_')))
        return params

    def delete(self, parameters):
        """
        Delete the given Random Forest variation. After the removal, a new default variation must be set. Raise a
        FileNotFoundError if the given parameters do not represent an existent variation.

        :param parameters: Number of trees
        :return: Nothing
        """
        # Check if the variation exists
        if not self.__check_variation_existence(parameters):
            raise FileNotFoundError
        # Loop over all directories (variations)
        names = [name for name in os.listdir(self.classifier_directory)]
        for name in names:
            check = True
            # If default it must take only the last parameter to compare
            if 'default' in name:
                params = name.split('_')
                for par_1, par_2 in zip(parameters, params[1:]):
                    if par_1 != par_2:
                        check = False
                        break
                if check:
                    shutil.rmtree(os.path.join(self.classifier_directory, name))
                    break
            # All others
            else:
                for par_1, par_2 in zip(parameters, name.split('_')):
                    if par_1 != par_2:
                        check = False
                        break
                if check:
                    shutil.rmtree(os.path.join(self.classifier_directory, name))
                    break

    def add(self, parameters):
        """
        Add a new Random Forest variation to the system. The new variation is set as default automatically. If the
        variation already exists raise a FileExistsError. If the parameters are not of the right format raise a
        AttributeError.

        :param parameters:  Number of trees
        :return: Nothing
        """
        if len(parameters) != 1 or not self.__check_int(parameters[0]):
            raise AttributeError
        if self.__check_variation_existence(parameters):
            raise FileExistsError
        else:
            name = '_'.join(parameters)
            os.makedirs(os.path.join(self.classifier_directory, name))
            self.set_default(parameters)

    def set_default(self, parameters):
        """
        Set as default the given Random Forest variation. Raise a FileNotFoundError if the given parameters do not
        represent an existent variation.

        :param parameters: Number of trees
        :return: Nothing
        """
        # Check if the variation exists
        if not self.__check_variation_existence(parameters):
            raise FileNotFoundError
        # Loop over all folder
        names = [name for name in os.listdir(self.classifier_directory)]
        for n in names:
            # Remove default
            if 'default' in n:
                check = False
                params = n.split('_')
                # Only if it is not the same as the given parameters
                for par_1, par_2 in zip(parameters, params[1:]):
                    if par_1 != par_2:
                        os.rename(os.path.join(self.classifier_directory, n),
                                  os.path.join(self.classifier_directory, '_'.join(params[1:])))
                        check = True
                        break
                if check:
                    continue
            # Set to default the new variation
            else:
                check = False
                for par_1, par_2 in zip(parameters, n.split('_')):
                    if par_1 != par_2:
                        check = True
                        break
                if check:
                    continue
                os.rename(os.path.join(self.classifier_directory, n),
                          os.path.join(self.classifier_directory, 'default_' + n))

    # ############################ HELPER ############################

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

    def __check_variation_existence(self, parameters):
        """
        Check if the given variation exists.

        :param parameters: Number of trees
        :return: True if the variation exist otherwise False
        """
        name = '_'.join(parameters)
        if os.path.exists(os.path.join(self.classifier_directory, name)) or os.path.exists(
                os.path.join(self.classifier_directory, 'default_' + name)):
            return True
        else:
            return False

    def __load_default(self):
        """
        Get parameters of the default implementation.

        :return: Parameters of the default implementation
        """
        names = [name for name in os.listdir(self.classifier_directory)]
        for name in names:
            if 'default' in name:
                params = name.split('_')
                # Return parameters without the keyword default
                return params[1:]
        return None

    def __get_default_name(self, name):
        """
        Get the directory name of the default variation of the given feature name.

        :param name: Name of the feature variation name
        :return: Name of the default classifier variation directory
        """
        return os.path.join(self.classifier_directory, 'default_' + '_'.join(self.__load_default()), name + ".out")
