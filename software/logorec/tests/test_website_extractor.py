import sys
import os

sys.path.append("../logorec")
from imageExtractor.websiteExtractor import WebsiteExtractor
import unittest


class WebsiteExtractorTestCase(unittest.TestCase):
    """
    Test Suite for the WebsiteExtractor
    """

    def setUp(self):
        """
        Set up components for each test

        :return: nothing
        """
        self.extractor = WebsiteExtractor()

    def tearDown(self):
        """
        Clean up after each test

        :return: nothing
        """
        if os.path.exists("data/imageExtractor/download"):
            self.extractor.clear()

    def test_extract(self):
        """
        Test that the website extractor achieves to download images from a website.

        :return: nothing
        """
        self.assertTrue(len(self.extractor.extract("http://www.google.com/", [1])) > 0)

    def test_wrong_website_extract(self):
        """
        Test website extraction error when the website given is not a valid URL.

        :return: nothing
        """
        with self.assertRaises(AttributeError):
            self.extractor.extract("thisisnotawebsite", [0])

    def test_to_much_parameters_website_extract(self):
        """
        Test website extract error when the parameters are wrong (to much parameters).

        :return: nothing
        """
        with self.assertRaises(ValueError):
            self.extractor.extract("validwebsite", [0, 0, 0])

    def test_wrong_parameters_website_extract(self):
        """
        Test website extract error when the parameters are wrong (to much parameters or wrong type).

        :return: nothing
        """
        with self.assertRaises(ValueError):
            self.extractor.extract("validwebsite", ['s'])


    def test_strange_link_website_extract(self):
        """
        Test that the website extractor achieves to download strange images from a website.

        :return: nothing
        """
        self.assertTrue(len(self.extractor.extract("https://shop.swatch.com/de_ch/", [0])) > 0)

    def test_clear(self):
        """
        Test that the website extractor achieve to delete the images downloaded

        :return: nothing
        """
        self.extractor.extract("http://www.google.com/", [0])
        self.extractor.clear()
        self.assertFalse(os.path.exists("data/imageExtractor/download"))
