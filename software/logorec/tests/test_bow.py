import sys

sys.path.append("../logorec")
from imageManager.directoryManager import DirectoryManager
from features.bow import Bow
from classifiers.randomForest import RandomForest
import unittest


class BowTestSuite(unittest.TestCase):
    """
    Test suite for the Bow Feature
    """

    def setUp(self):
        """
        Set up components for each test.

        :return: nothing
        """
        self.bow = Bow()
        self.bow.add(['10', '1', '1', "True"])
        self.manager = DirectoryManager()
        self.manager.add_category("test")
        self.manager.save("data/features/banner/logos_01.jpg", "test")
        self.manager.save("data/features/banner/logos_02.png", "test")
        self.manager.save("data/features/banner/logos_03.png", "test")
        self.manager.add_category("testtwo")
        self.manager.save("data/features/banner/logos_01.jpg", "testtwo")
        self.manager.save("data/features/banner/logos_02.png", "testtwo")
        self.manager.save("data/features/banner/logos_03.png", "testtwo")

    def tearDown(self):
        """
        Clean up after each test.

        :return: nothing
        """
        self.bow.delete(['10', '1', '1', "True"])
        self.manager.delete_category("test")
        self.manager.delete_category("testtwo")

    # def test_train_classifier_probability_and_services(self):
    #     """
    #     Test if the train process, get probability and get services work correctly.
    #     This test takes a lot of time to be excecuted (training processes)
    #
    #     :return: nothing
    #     """
    #     # this test also train and is_trained
    #     self.bow.train(self.manager.get_all())
    #     self.assertTrue(self.bow.is_trained())
    #
    #     classifier = RandomForest()
    #     classifier.add(['10'])
    #     self.bow.train_classifier(None, self.manager.generate_targets(), classifier)
    #     self.assertTrue(classifier.is_trained())
    #
    #     self.assertTrue(self.bow.probability(self.manager.get_all(), classifier) >= 0)
    #     self.assertTrue(self.bow.services(self.manager.get_all(), classifier)[0] >= 0 and len(
    #         self.bow.services(self.manager.get_all(), classifier)) == 2)
    #     classifier.delete(['10'])

    def test_need_train(self):
        """
        Test if the bow implementation need a train process

        :return: nothing
        """
        self.assertTrue(self.bow.need_train())

    def test_show(self):
        """
        Test if the show method return the right quantity of variations.

        :return: nothing
        """
        self.assertTrue(len(self.bow.show()) == 1)

    def test_fail_delete(self):
        """
        Test fail delete when the variation given does not exist.

        :return: nothing
        """
        with self.assertRaises(FileNotFoundError):
            self.bow.delete(['1', '1', '1', "True"])

    def test_delete(self):
        """
        Test if the delete variation deletes correctly.

        :return: nothing
        """
        self.bow.add(['2', '2', '2', "True"])
        self.bow.delete(['2', '2', '2', "True"])
        self.assertTrue(len(self.bow.show()) == 1)

    def test_add(self):
        """
        Test if add variation add correctly.
        
        :return: nothing
        """
        self.bow.add(['2', '2', '2', "True"])
        variations = self.bow.show()
        self.bow.delete(['2', '2', '2', "True"])
        self.assertTrue(len(variations) == 2)

    def test_exist_add(self):
        """
        Test fail add when the variation given does already exist.

        :return: nothing
        """
        with self.assertRaises(FileExistsError):
            self.bow.add(['10', '1', '1', "True"])

    def test_wrong_parameters_add(self):
        """
        Test add with wrong parameters (wrong number of parameters and wrong types)

        :return: nothing
        """
        with self.assertRaises(AttributeError):
            self.bow.add(['10', '1', '1', "True", 20])
            self.bow.add(['s', '1', '1', "True"])
            self.bow.add(['10', 's', '1', "True"])
            self.bow.add(['10', '1', 's', "True"])
            self.bow.add(['10', '1', '1', 's'])

    def test_fail_set_default(self):
        """
        Test fail set default when the variation given does not exist.

        :return: nothing
        """
        with self.assertRaises(FileNotFoundError):
            self.bow.set_default(['1', '1', '1', "True"])

    def test_set_default(self):
        """
        Test if the set default variation work correctly.

        :return: nothing
        """
        self.bow.add(['1', '1', '1', "True"])
        self.bow.set_default(['10', '1', '1', "True"])
        self.bow.delete(['1', '1', '1', "True"])
        self.assertTrue(self.bow.default_exist())

    def test_negative_is_trained(self):
        """
        Test if the is trained work correctly when the feature is not trained.

        :return: nothing
        """
        self.assertFalse(self.bow.is_trained())

    def test_positive_default_exist(self):
        """
        Test if the default variation exist work correctly.

        :return: nothing
        """
        self.assertTrue(self.bow.default_exist())

    def test_negative_dafault_exist(self):
        """
        Test negative default variation exist when there is any default variations.

        :return: nothing
        """
        self.bow.add(['1', '1', '1', "True"])
        self.bow.delete(['1', '1', '1', "True"])
        self.assertFalse(self.bow.default_exist())
