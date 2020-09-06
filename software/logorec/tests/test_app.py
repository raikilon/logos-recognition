import sys

sys.path.append("../logorec")
from app import App
import unittest


class AppTestSuite(unittest.TestCase):
    """
    Test suite for the main app.
    """

    def setUp(self):
        """
        Set up componenets for each test.

        :return: nothing
        """
        self.app = App()
        self.app.add_category("test")
        self.app.add_image("data/features/banner/logos_01.jpg", "test")

    def tearDown(self):
        """
        Clean up after each test.

        :return: nothing
        """
        self.app.delete_category("test")

    def test_feature_need_train(self):
        """
        Test if the need train works correctly.

        :return: nothing
        """
        self.assertTrue(self.app.feature_need_train("Bow"))

    def test_get_categories(self):
        """
        Test if the get categories works correctly.

        :return: nothing
        """
        self.assertTrue(len(self.app.get_categories()) == 1)

    def test_get_features(self):
        """
        Test if the get features works correctly.

        :return: nothing
        """
        self.assertTrue("Bow" in self.app.get_features())

    def test_get_classifiers(self):
        """
        Test if the get classifiers works correctly.

        :return: nothing
        """
        self.assertTrue("RandomForest" in self.app.get_classifiers())

    def test_get_default_classifier(self):
        """
        Test if the get default classifier works correctly.

        :return: nothing
        """
        self.assertTrue(self.app.get_default_classifier() in self.app.get_classifiers())

    def test_get_default_feature(self):
        """
        Test if the get default feature works correctly.

        :return: nothing
        """
        self.assertTrue(self.app.get_default_feature() in self.app.get_features())

    def test_error_get_probability(self):
        """
        Test error get probability when the classifier is not trained.

        :return: nothing
        """
        with self.assertRaises(ModuleNotFoundError):
            self.app.get_probability("website", [1], "RandomForest", "Bow")

    def test_error_get_services(self):
        """
        test error get services when the classifier is not trained.

        :return: nothing
        """
        with self.assertRaises(ModuleNotFoundError):
            self.app.get_services("website", [1], "RandomForest", "Bow")

    def test_fail_train_feature(self):
        """
        Test fail train feature when a default feature's variation does not exist.

        :return: nothing
        """
        with self.assertRaises(ModuleNotFoundError):
            self.app.train_feature("Bow")

    def test_fail_train_classifier(self):
        """
        Test fail train classifier when a default classifier' variation does not exist.

        :return: nothing
        """
        with self.assertRaises(ModuleNotFoundError):
            self.app.train_classifier("RandomForest", "Bow")

    def test_add_category(self, ):
        """
        Test if add category works correctly.

        :return: nothing
        """
        self.app.add_category("testthree")
        categories = self.app.get_categories()
        self.app.delete_category("testthree")
        self.assertTrue("testthree" in categories)

    def test_add_image(self):
        """
        Test if add image works correctly.

        :return: nothing
        """
        self.app.add_image("data/features/banner/logos_02.png", "test")
        self.assertTrue(any("logos_02" in i for i in self.app.show_images_by_category("test")))

    def test_add_classifier(self):
        """
        Test add classifier's variation works correctly.

        :return: nothing
        """
        self.app.add_classifier("RandomForest", ['2'])
        variations = self.app.show_classifier_variations("RandomForest")
        self.app.delete_classifier("RandomForest", ["2"])
        self.assertTrue(len(variations) == 1)

    def test_add_feature(self):
        """
        Test if add feature's variation works correctly.

        :return: nothing
        """
        self.app.add_feature("Bow", ['2', '2', '2', "True"])
        variations = self.app.show_feature_variations("Bow")
        self.app.delete_feature("Bow", ['2', '2', '2', "True"])
        self.assertTrue(len(variations) == 1)

    def test_show_images_by_category(self):
        """
        Test if show image by category return the right amount of images.

        :return: nothing
        """
        self.assertTrue(len(self.app.show_images_by_category("test")) == 1)

    def test_delete_image_by_category(self):
        """
        Test if delete image by category deletes correctly.

        :return: nothing
        """
        self.app.add_image("data/features/banner/logos_02.png", "test")
        self.app.delete_image_by_category("test", "logos_02.png")
        self.assertTrue(len(self.app.show_images_by_category("test")) == 1)

    def test_delete_category(self):
        """
        Test if delete category work correctly.

        :return: nothing
        """
        self.app.add_category("testthree")
        self.app.delete_category("testthree")
        self.assertFalse("testthree" in self.app.get_categories())

    def test_set_default_classifier(self):
        """
        Test if set default classifier's variation work correctly.

        :return: nothing
        """
        self.app.add_classifier("RandomForest", ['2'])
        self.app.add_classifier("RandomForest", ['1'])
        self.app.delete_classifier("RandomForest", ['1'])
        self.app.set_default_classifier("RandomForest", ['2'])
        default = self.app.classifier.default_exist()
        self.app.delete_classifier("RandomForest", ['2'])
        self.assertTrue(default)

    def test_set_default_feature(self):
        """
        Test if set default feature's variation work correctly.

        :return: nothing
        """
        self.app.add_feature("Bow", ['10', '1', '1', 'True'])
        self.app.add_feature("Bow", ['1', '1', '1', 'True'])
        self.app.delete_feature("Bow", ['1', '1', '1', 'True'])
        self.app.set_default_feature("Bow", ['10', '1', '1', 'True'])
        default = self.app.feature.default_exist()
        self.app.delete_feature("Bow", ['10', '1', '1', 'True'])
        self.assertTrue(default)
