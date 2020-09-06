from lib.imageExtractor import ImageExtractor
import os


class TestExtractor(ImageExtractor):

    def __init__(self):
        self.banner_dir = 'data/features/banner/'

    def extract(self, website, parameters):
        """
        Return all the images used for the hard negative update.

        :param website: None
        :param parameters: None
        :return: List of images paths.
        """
        images = []
        for k, dir_name in enumerate(os.listdir(self.banner_dir)):
            images.append(os.path.join(self.banner_dir, dir_name))

        return images

    def clear(self):
        """
        Delete all the extracted images.

        :return: Nothing
        """
        pass
