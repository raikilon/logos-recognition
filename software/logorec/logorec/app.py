import configparser
import logging
import os


class App(object):

    def __init__(self):
        """
        Initiate the different parameters, i.e. import all the default settings from the parameters.ini file.

        """
        self.feature = None
        self.classifier = None
        # Instantiate Logging (change to debug to see messages)
        logging.basicConfig(level=logging.FATAL)
        logging.debug("App instantiation")

        # Import parameters
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parameters.ini'))
        self.default_classifier = self.config.get('DEFAULT', 'ClassifierFile')
        self.default_feature = self.config.get('DEFAULT', 'FeatureFile')
        # Import module and classes

        # Load ImageExtractor
        # To compile doc change the package to logorec.imageExtractor
        module = __import__("imageExtractor." + self.__lowercase(self.config.get('DEFAULT', 'ImageExtractorFile')),
                            fromlist=['imageExtractor'])
        class_ = getattr(module, self.config.get('DEFAULT', 'ImageExtractorFile'))
        self.image_extractor = class_()

        # Load ImageManager
        # To compile doc change the package to logorec.imageManager
        module = __import__("imageManager." + self.__lowercase(self.config.get('DEFAULT', 'ImageManagerFile')),
                            fromlist=['imageManager'])
        class_ = getattr(module, self.config.get('DEFAULT', 'ImageManagerFile'))
        self.image_manager = class_()

    # ############################ HELPER ############################

    @staticmethod
    def __lowercase(s):
        """
        Lowercase the first letter of the given string.

        :param s: String to lowercase
        :return: String with lowercase first letter
        """
        if s:
            return s[:1].lower() + s[1:]
        else:
            return ''

    @staticmethod
    def __uppercase(s):
        """
        Uppercase the first letter of the given string.

        :param s: String to uppercase
        :return: String with uppercase first letter
        """
        if s:
            return s[:1].upper() + s[1:]
        else:
            return ''

    def __load_classifier(self, classifier=None):
        """
        Set given classifier as current (self.classifier). If the classifier is not given the default one is set as current.

        :param classifier: Name of the classifier class (i.e. RandomForest)
        :return: Nothing
        """
        # Load given classifier if it is not None
        if classifier is not None:
            module = __import__("classifiers." + self.__lowercase(classifier),
                                fromlist=['classifiers'])
            class_ = getattr(module, classifier)
        # Load default classifiers
        else:
            module = __import__("classifiers." + self.__lowercase(self.config.get('DEFAULT', 'ClassifierFile')),
                                fromlist=['classifiers'])
            class_ = getattr(module, self.config.get('DEFAULT', 'ClassifierFile'))
        self.classifier = class_()

    def __load_feature(self, feature=None):
        """
        Set given feature as current (self.feature). If the feature is not given the default one is set as current.

        :param feature: Name of the feature class (i.e. Bow)
        :return: Nothing
        """
        # Load given feature if it is not None
        if feature is not None:
            module = __import__("features." + self.__lowercase(feature),
                                fromlist=['features'])
            class_ = getattr(module, feature)
        # Load default feature
        else:
            module = __import__("features." + self.__lowercase(self.config.get('DEFAULT', 'FeatureFile')),
                                fromlist=['features'])
            class_ = getattr(module, self.config.get('DEFAULT', 'FeatureFile'))
        self.feature = class_()

    def feature_need_train(self, feature):
        """
        Check if the given feature need a training phase.

        :param feature:  Name of the feature class (i.e. Bow)
        :return: True if the feature need a train otherwise False
        """
        self.__load_feature(feature)
        return self.feature.need_train()

    def get_categories(self):
        """
        Retrieve all the logo categories present in the default ImageManager.

        :return: List of logo categories
        """
        logging.debug("get categories")
        return self.image_manager.get_categories()

    def get_features(self):
        """
        Retrieve all the feature algorithm present in the library.

        :return: List of feature algorithms
        """
        logging.debug("get features")
        feature = []
        for file in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "features")):
            if file.endswith(".py") and not file == '__init__.py' and not file == '__pycache__':
                filename, file_extension = os.path.splitext(file)
                feature.append(self.__uppercase(filename))
        return feature

    def get_classifiers(self):
        """
        Retrieve all the classifier present in the library.

        :return: List of classifiers
        """
        logging.debug("get classifiers")
        classifier = []
        for file in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "classifiers")):
            if file.endswith(".py") and not file == '__init__.py' and not file == '__pycache__':
                filename, file_extension = os.path.splitext(file)
                classifier.append(self.__uppercase(filename))
        return classifier

    def get_default_classifier(self):
        """
        Retrieve the default classifier from the parameters.ini.

        :return: Default classifier
        """
        logging.debug("get default classifier")
        return self.default_classifier

    def get_default_feature(self):
        """
        Retrieve the default feature from the parameters.ini.

        :return: Default feature
        """
        logging.debug("get default feature")
        return self.default_feature

    # ############################ RETRIEVE INFORMATION ############################

    def get_probability(self, website, parameters, classifier, feature):
        """
        Get the probability that the given website is a web shop. The given website, classifier and feature must exist!
        If the parameters do not correspond to the ImageExtractor implementation a ValueError is raised. If the
        classifier is not trained a ModuleNotFoundError is raised. If files for the classification are missing a
        FileNotFoundError is raised. If the given websites do not correspond with the implementation a AttributeError is
        raised.

        :param website: Website url (i.e. http://www.google.com)
        :param parameters: Parameter for the ImageExtractor. See specific implementation doc.
        :param classifier: Classifier to use for the classification phase (None to use the default one)
        :param feature: Feature algorithm to extract information from website's images
        :return: Probability (0-100) that the given website is a web shop
        """
        logging.debug("getting probability")
        self.__load_feature(feature)
        self.__load_classifier(classifier)
        if not self.classifier.is_trained():
            raise ModuleNotFoundError
        images = self.image_extractor.extract(website, parameters)
        if images:
            probs = self.feature.probability(images, self.classifier)
            self.image_extractor.clear()
        else:
            probs = 0
        return probs

    def get_services(self, website, parameters, classifier, feature):
        """
        Get the services offered by the given website. The given website, classifier and feature must exist! If the
        parameters do not correspond to the ImageExtractor implementation a ValueError is raised. If the classifier is
        not trained a ModuleNotFoundError is raised. If files for the classification are missing a FileNotFoundError is
        raised. If the given websites do not correspond with the implementation a AttributeError is raised.

        :param website: Website url (i.e. http://www.google.com)
        :param parameters: Parameter for the ImageExtractor. See specific implementation doc.
        :param classifier: Classifier to use for the classification phase (None to use the default one)
        :param feature: Feature algorithm to extract information from website's images
        :return: List of services probability (0-100). The number of probability is the number of available categories
        """
        logging.debug("getting probability")
        self.__load_feature(feature)
        self.__load_classifier(classifier)
        if not self.classifier.is_trained():
            raise ModuleNotFoundError
        images = self.image_extractor.extract(website, parameters)
        if images:
            services = self.feature.services(images, self.classifier)
            self.image_extractor.clear()
        else:
            services = len(self.get_categories()) * [0]
        return services

    # ############################ TRAIN OPERATIONS ############################

    def train_feature(self, feature):
        """
        Train the given feature. Check before that the feature need a training phase with feature_need_train(). The
        given feature must exist! If the given feature type does not have a default implementation a ModuleNotFoundError
        is raised.

        :param feature: Feature algorithm to train
        :return: Nothing
        """
        logging.debug("training feature")
        self.__load_feature(feature)
        if self.feature.default_exist():
            self.feature.train(self.image_manager.get_all())
        else:
            raise ModuleNotFoundError

    def train_classifier(self, classifier, feature):
        """
        Train the given classifier with the given feature. If the feature is None the default one is used. If the given
        feature need a train, it must be trained before the classifier otherwise, a ModuleNotFoundError is raised. If
        the classifier does not have a default implementation a ModuleNotFoundError is raised.

        :param classifier: Classifier to train
        :param feature: Feature to train the classifier
        :return: Nothing
        """
        logging.debug("training classifier")
        self.__load_feature(feature)
        self.__load_classifier(classifier)
        if ((self.feature.need_train() and self.feature.is_trained())
            or not self.feature.need_train()) and self.classifier.default_exist():
            self.feature.train_classifier(self.image_manager.get_all(),
                                          self.image_manager.generate_targets(),
                                          self.classifier)
        else:
            raise ModuleNotFoundError

    # ############################ ADD OPERATIONS ############################

    def add_category(self, category):
        """
        Add a new logo category with the default ImageManager. The given category must exist!

        :param category: Name of the new logo category
        :return: Nothing
        """
        logging.debug("add category")
        self.image_manager.add_category(category)

    def add_image(self, image, category):
        """
        Add a new image to the given logo category using the default ImageManager. The given category and image must
        exist!

        :param image: Image path to the new image
        :param category: Logo category name
        :return: Nothing
        """
        logging.debug("saving image")
        self.image_manager.save(image, category)

    def add_classifier(self, classifier, parameters):
        """
        Add a new classifier variation. The given classifier must exist! If the classifier variation already exists a
        FileExistsError is raised. If parameters don't correspond with the implementation a AttributeError is raised.

        :param classifier: Classifier type
        :param parameters: Parameters for the classifier. Check specific implementation doc.
        :return: Nothing
        """
        logging.debug("adding classifier")
        self.__load_classifier(classifier)
        self.classifier.add(parameters)

    def add_feature(self, feature, parameters):
        """
        Add a new feature variation. The given feature must exist! If the feature variation already exists a
        FileExistsError is raised. If parameters do not correspond with the implementation a AttributeError is raised.

        :param feature: Feature type
        :param parameters: Parameters for the Feature. Check specific implementation doc.
        :return: Nothing
        """
        logging.debug("adding feature")
        self.__load_feature(feature)
        self.feature.add(parameters)

    # ############################ SHOW OPERATIONS ############################

    def show_images_by_category(self, category):
        """
        Show all the images of a given category of logo. The given category must exist!

        :param category: Logo category name
        :return: List of images paths. If there are not images the list is empty.
        """
        logging.debug("showing images in category")
        return self.image_manager.get_by_category(category)

    def show_classifier_variations(self, classifier):
        """
        Show all the classifier variations. The given classifier must exist!

        :param classifier: Classifier name
        :return: List of classifier parameters. If there are not variations the list is empty.
        """
        logging.debug("showing classifier variations")
        self.__load_classifier(classifier)
        return self.classifier.show()

    def show_feature_variations(self, feature):
        """
        Show all the feature variations. The given feature must exist!

        :param feature: Feature name
        :return: List of feature parameters. If there are not variations the list is empty.
        """
        logging.debug("showing feature variations")
        self.__load_feature(feature)
        return self.feature.show()

    # ############################ DELETE OPERATIONS ############################

    def delete_image_by_category(self, category, image):
        """
        Delete the image from the given logo category set. The given category must exist! If the image does not exist a
        FileNotFoundError is raised.

        :param category: Name of the logo category
        :param image: Name of the image
        :return: Nothing
        """
        logging.debug("deleting image")
        self.image_manager.delete_by_category(category, image)

    def delete_category(self, category):
        """
        Delete the given category and all images contained in it. The given category must exist!

        :param category: Name of the logo category
        :return: Nothing
        """
        logging.debug("deleting category")
        self.image_manager.delete_category(category)

    def delete_classifier(self, classifier, parameters):
        """
        Delete the variation of the given classifier. The given classifier must exist! If the parameters do not
        represent any classifier variations a FileNotFoundError is raised.

        :param classifier: Name of the classifier
        :param parameters: Parameters of the classifier variation
        :return: Nothing
        """
        logging.debug("deleting classifier")
        self.__load_classifier(classifier)
        self.classifier.delete(parameters)

    def delete_feature(self, feature, parameters):
        """
        Delete the variation of the given feature. The given feature must exist! If the parameters do not represent
        any feature variations a FileNotFoundError is raised.

        :param feature: Name of the classifier
        :param parameters: Parameters of the feature variation
        :return: Nothing
        """
        logging.debug("deleting feature")
        self.__load_feature(feature)
        self.feature.delete(parameters)

    # ############################ SET OPERATIONS ############################

    def set_default_classifier(self, classifier, parameters):
        """
        Set the variation of the given classifier as default. The given classifier must exist! If the parameters do not
        represent any classifier variations a FileNotFoundError is raised.

        :param classifier: Name of the classifier
        :param parameters: Parameters of the classifier variation
        :return: Nothing
        """
        logging.debug("setting default classifier")
        self.__load_classifier(classifier)
        self.classifier.set_default(parameters)

    def set_default_feature(self, feature, parameters):
        """
        Set the variation of the given feature as default. The given feature must exist! If the parameters do not
        represent any feature variations a FileNotFoundError is raised.

        :param feature: Name of the classifier
        :param parameters: Parameters of the feature variation
        :return: Nothing
        """
        logging.debug("setting default feature")
        self.__load_feature(feature)
        self.feature.set_default(parameters)


if __name__ == '__main__':
    app = App()
