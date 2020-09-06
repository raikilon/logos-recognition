from lib.imageManager import ImageManager
import os
import shutil


class DirectoryManager(ImageManager):
    """
    This class allows to manage all the categories with directory on the local disk.
    """

    def __init__(self):
        """
        Initiate the directory manager by creating the directory in data/categories.
        """
        # data directory
        self.categories_director = 'data/categories'
        # Generate data directory if it does not exist
        if not os.path.exists(self.categories_director):
            os.makedirs(self.categories_director)

    def get_all(self):
        """
        Get all the images paths.

        :return: List of images absolute paths
        """
        paths = []
        # For each category ask for all its images paths
        for cat in self.get_categories():
            images = self.get_by_category(cat)
            # Concat all the images in the same list
            paths.extend(images)
        return paths

    def get_categories(self):
        """
        Get all categories.

        :return: List of all categories names
        """
        return [name for name in os.listdir(self.categories_director)]

    def get_by_category(self, category):
        """
        Get images path for the given category.

        :param category: Image category
        :return: List of all images absolute paths
        """
        return [os.path.join(os.path.abspath(self.categories_director), category, f) for f in
                os.listdir(os.path.join(self.categories_director, category)) if
                os.path.isfile(os.path.join(self.categories_director, category, f))]

    def add_category(self, category):
        """
        Add a new category.

        :param category: Category name
        :return: Nothing
        """
        os.makedirs(os.path.join(self.categories_director, category))

    def delete_category(self, category):
        """
        Delete category and all its content.

        :param category: Category name
        :return: Nothing
        """
        shutil.rmtree(os.path.join(self.categories_director, category))

    def delete_by_category(self, category, image):
        """
        Delete image in the given category. If the image does not exist raise FileNotFoundError.

        :param category: Category name
        :param image: Image name
        :return: Nothing
        """
        if os.path.isfile(os.path.join(self.categories_director, category, image)):
            os.remove(os.path.join(self.categories_director, category, image))
        else:
            raise FileNotFoundError

    def save(self, image, category):
        """
        Add new image to the given category. If the given image's name already exists the image is overwritten.

        :param image: Image absolute path
        :param category: Category name
        :return: Nothing
        """
        shutil.copy(image, self.categories_director + '/' + category)

    def generate_targets(self, number_of_classes=True):
        """
        Generate targets in the same order of the get_all().

        :param number_of_classes: True (default) if the must be only two classes.  True if there is a class for each category.
        :return: List of classes (if number_of_classes is True the list contain another list of reach category)
        """
        targets = []
        if number_of_classes:
            for cat in self.get_categories():
                classes = []
                for c in self.get_categories():
                    if c != cat:
                        classes += [1] * len(self.get_by_category(c))
                    else:
                        classes += [0] * len(self.get_by_category(c))
                targets.append(classes)
        else:
            for count, cat in enumerate(self.get_categories()):
                targets.extend([count] * len(self.get_by_category(cat)))
        return targets
