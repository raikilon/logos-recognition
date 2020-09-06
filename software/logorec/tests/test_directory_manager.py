import sys

sys.path.append("../logorec")
from imageManager.directoryManager import DirectoryManager
import unittest


class DirectoryManagerTestCase(unittest.TestCase):
    """
    Test suit for directory manager.
    """

    def setUp(self):
        """
        Set up components for each test.

        :return: nothing
        """
        self.manager = DirectoryManager()
        self.manager.add_category("test")
        self.manager.save("data/features/banner/logos_01.jpg", "test")

    def tearDown(self):
        """
        Clean up after each test.

        :return: nothing
        """
        self.manager.delete_category("test")

    def test_get_all(self):
        """
        Test if get all returns the right images.

        :return: nothing
        """
        self.assertTrue(any("logos_01" in i for i in self.manager.get_all()))

    def test_get_categories(self):
        """
        Test if get categories returns the right categories.

        :return: nothing
        """
        self.assertTrue("test" in self.manager.get_categories())

    def test_get_by_category(self):
        """
        Test if get images by category  returns the right images.

        :return: nothing
        """
        self.assertTrue(any("logos_01" in i for i in self.manager.get_by_category("test")))

    def test_add_category(self):
        """
        Test if add category is added correctly to the system.

        :return: nothing
        """
        self.manager.add_category("testtwo")
        categories = self.manager.get_categories()
        self.manager.delete_category("testtwo")
        self.assertTrue("testtwo" in categories)

    def test_delete_category(self):
        """
        Test if delete category deletes correctly a category from the system.

        :return: nothing
        """
        self.manager.add_category("testtwo")
        self.manager.delete_category("testtwo")
        self.assertFalse("testtwo" in self.manager.get_categories())

    def test_delete_by_category(self):
        """
        Test of delete images by category delete the right image in the right category.

        :return: nothing
        """
        self.manager.delete_by_category("test", "logos_01.jpg")
        self.assertFalse(any("logos_01" in i for i in self.manager.get_by_category("test")))

    def test_exception_delete_by_category(self):
        """
        Check fail delete by category when the image given does not exist.

        :return: nothing
        """
        with self.assertRaises(FileNotFoundError):
            self.manager.delete_by_category("test", "noexist.jpg")

    def test_save(self):
        """
        Test if the save image saves correctly an image in the category given.

        :return: nothing
        """
        self.manager.save("data/features/banner/logos_02.png", "test")
        self.assertTrue(any("logos_02" in i for i in self.manager.get_by_category("test")))

    def test_generate_targets(self):
        """
        Test if generates targets generate the right data for the given images.

        :return: nothing
        """
        self.assertTrue(len(self.manager.generate_targets()) > 0 and len(self.manager.generate_targets(False)) > 0)
