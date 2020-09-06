import sys

sys.path.append("../logorec")
from classifiers.randomForest import RandomForest
import unittest


class RandomForestTestSuite(unittest.TestCase):
    """
    Test suite for the Random Forest classifier
    """

    def setUp(self):
        """
        Set up componenets for each test
        :return:
        """
        self.classifier = RandomForest()
        self.classifier.add(['10'])

    def tearDown(self):
        """
        Clean up after each test

        :return: nothing
        """
        self.classifier.delete(['10'])

    def test_negative_is_trained(self):
        """
        Test negative is trained when the classifier is not trained.

        :return: nothing
        """
        self.assertFalse(self.classifier.is_trained())

    def test_positive_default_exist(self):
        """
        Test positive default exist when there is a default variation in the system.

        :return: nothing
        """
        self.assertTrue(self.classifier.default_exist())

    def test_negative_dafault_exist(self):
        """
        Test negative default exist when there is not a default variation in the system.

        :return: nothing
        """
        self.classifier.add(['1'])
        self.classifier.delete(['1'])
        self.assertFalse(self.classifier.default_exist())

    def test_show(self):
        """
        Test the show method. Check if it return the right amount of variations.

        :return: nothing
        """
        self.assertTrue(len(self.classifier.show()) == 1)

    def test_fail_delete(self):
        """
        Test fail delete when the variation to delete does not exist.

        :return: nothing
        """
        with self.assertRaises(FileNotFoundError):
            self.classifier.delete(['1'])

    def test_delete(self):
        """
        Test delete when the variation to delete does exist.

        :return: nothing
        """
        self.classifier.add(['2'])
        self.classifier.delete(['2'])
        self.assertTrue(len(self.classifier.show()) == 1)

    def test_add(self):
        """
        Test add a new variation to the classifier.

        :return: nothing
        """
        self.classifier.add(['2'])
        variations = self.classifier.show()
        self.classifier.delete(['2'])
        self.assertTrue(len(variations) == 2)

    def test_exist_add(self):
        """
        Test fail add when the new variation already exists.

        :return: nothing
        """
        with self.assertRaises(FileExistsError):
            self.classifier.add(['10'])

    def test_wrong_parameters_add(self):
        """
        Test add with wrong parameters (to much parameters and wrong type parameter)

        :return: nothing
        """
        with self.assertRaises(AttributeError):
            self.classifier.add(['10', '1'])
            self.classifier.add(['s'])

    def test_fail_set_default(self):
        """
        Test fail set default when the default variation does not exist.

        :return: nothing
        """
        with self.assertRaises(FileNotFoundError):
            self.classifier.set_default(['1'])

    def test_set_default(self):
        """
        Test set default variation to the classifier.

        :return: nothing
        """
        self.classifier.add(['1'])
        self.classifier.set_default(['10'])
        self.classifier.delete(['1'])
        self.assertTrue(self.classifier.default_exist())
