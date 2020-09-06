import click
import urllib.request
from app import App

# ############################ PARAMETERS ############################
app = App()
feature_list = app.get_features()
classifier_list = app.get_classifiers()
categories_list = app.get_categories()
default_feature = app.get_default_feature()
default_classifier = app.get_default_classifier()


# ############################ CALLBACKS ############################

def validate_category_existence(ctx, param, value):
    """
    Validate that the given category does not already exist.
    """
    if isinstance(value, str):
        value = value.upper()
        if value not in categories_list:
            return value
        else:
            raise click.BadParameter("The given category already exists")
    else:
        raise click.UsageError("The given category is not text")


def validate_feature_train(ctx, param, value):
    """
    Validate if the feature need a training phase.
    """
    if app.feature_need_train(value):
        return value
    else:
        raise click.UsageError("The given feature type does not need a training phase")


# ############################ MAIN CLICK ############################

@click.group()
def click1():
    """
    This is the main commands group.
    """


# ############################ RETRIEVE INFORMATION ############################

@click1.group()
def get():
    """
    Retrieve the website's probability to be a web shop,
    to discover the website's services.
    """
    pass


@get.command()
@click.argument('websites', required=True, nargs=-1)
@click.option('--parameters', '-p', multiple=True, default=None)
@click.option('--feature', '-f', default=None, type=click.Choice(feature_list),
              help="Feature generation algorithm (default {0})".format(default_feature))
@click.option('--classifier', '-c', default=default_classifier, type=click.Choice(classifier_list),
              help="Classifier type (default {0})".format(default_classifier))
def probability(websites, parameters, feature, classifier):
    """
    Retrieve the probability that the given website is a web shop. Multiple websites URLs can be given.
    """
    for website in websites:
        try:
            prob = app.get_probability(website, parameters, classifier, feature)
            click.echo("{0} is a web shop with a probability of {1:2.2f} %".format(website, prob))
        except AttributeError:
            raise click.BadParameter("Websites do not correspond with the implementation")
        except ValueError:
            raise click.BadParameter("Parameters do not correspond with the implementation")
        except ModuleNotFoundError:
            raise click.FileError("Classifier is not trained")
        except FileNotFoundError:
            raise click.FileError("Classifier is not trained correctly")


@get.command()
@click.argument('websites', required=True, nargs=-1)
@click.option('--parameters', '-p', multiple=True, default=None)
@click.option('--features', '-f', default=None, type=click.Choice(feature_list),
              help="Feature generation algorithm (default {0})".format(default_feature))
@click.option('--classifier', '-c', default=default_classifier, type=click.Choice(classifier_list),
              help="Classifier type (default {0})".format(default_classifier))
def services(websites, parameters, feature, classifier):
    """
    Retrieve the name of the different services offered by the given website. Multiple website URLs can be given.
    """
    for website in websites:
        try:
            probs = app.get_services(website, parameters, classifier, feature)
            for cat, prob in zip(categories_list, probs):
                click.echo("{0} is a offered by {1} with a probability of {2:2.2f} %".format(cat, website, prob))
        except AttributeError:
            raise click.BadParameter("Websites do not correspond with the implementation")
        except ValueError:
            raise click.BadParameter("Parameters do not correspond with the implementation")
        except ModuleNotFoundError:
            raise click.FileError("Classifier is not trained")
        except FileNotFoundError:
            raise click.FileError("Classifier is not trained correctly")


# ############################ TRAIN INFORMATION ############################

@click1.group()
def train():
    """
    Train the classifier or the feature model.
    """
    pass


@train.command()
@click.argument('type', type=click.Choice(feature_list), callback=validate_feature_train)
@click.confirmation_option(prompt='Are you sure you want to train ? The process could require a lot of time.')
def feature(type):
    """
    Train the given feature TYPE with the default parameters.
    """
    try:
        app.train_feature(type)
    except ModuleNotFoundError:
        raise click.FileError("The used feature type does not have a default implementation")


@train.command()
@click.argument('type', type=click.Choice(classifier_list))
@click.option('--feature', '-f', default=None, type=click.Choice(feature_list),
              help="Feature generation algorithm (default {0})".format(default_feature))
@click.confirmation_option(prompt='Are you sure you want to train ? The process could require a lot of time?')
def classifier(type, feature):
    """
    Train the given classifier TYPE with the given parameters.
    """
    try:
        app.train_classifier(type, feature)
    except ModuleNotFoundError:
        raise click.FileError(
            "The used feature must be trained or the classifier does not have a default implementation")


# ############################ ADD OPERATIONS ############################

@click1.group()
def add():
    """
    Add new images, categories, classifiers, features, etc.
    """
    pass


@add.command()
@click.argument('category', callback=validate_category_existence)
def category(category):
    """
    Add a new CATEGORY to the system.
    """
    app.add_category(category)


@add.command()
@click.argument('category', type=click.Choice(categories_list), nargs=1)
@click.argument('images', required=True, type=click.Path(exists=True), nargs=-1)
def image(category, images):
    """
    Add the given IMAGES to a given CATEGORY set
    """
    for img in images:
        app.add_image(img, category)


@add.command()
@click.argument('type', type=click.Choice(classifier_list), nargs=1)
@click.argument('parameters', required=True, nargs=-1)
def classifier(type, parameters):
    """
    Add the classifier with the given PARAMETERS among those of the given TYPE.
    """
    try:
        app.add_classifier(type, parameters)
    except AttributeError:
        raise click.BadParameter("Parameters do not correspond with the classifier implementation")
    except FileExistsError:
        raise click.FileError("The classifier variation already exists")


@add.command()
@click.argument('type', type=click.Choice(feature_list), nargs=1)
@click.argument('parameters', required=True, nargs=-1)
def feature(type, parameters):
    """
    Add the feature with the given PARAMETERS among those of the given TYPE.
    """
    try:
        app.add_feature(type, parameters)
    except AttributeError:
        raise click.BadParameter("Parameters do not correspond with the feature implementation")
    except FileExistsError:
        raise click.FileError("The feature variation already exists")


# ############################ SHOW OPERATIONS ############################

@click1.group()
def show():
    """
    Shows images, classifiers, features, etc.
    """
    pass


@show.command()
@click.argument('category', type=click.Choice(categories_list))
def images(category):
    """
    Show all the images' names contained in the given CATEGORY set.
    """
    images = app.show_images_by_category(category)
    click.echo_via_pager('\n'.join(images))


@show.command()
@click.argument('type', type=click.Choice(classifier_list))
def classifier(type):
    """
    Show all the classifiers variation of the given TYPE.
    """
    classifier = app.show_classifier_variations(type)
    click.echo_via_pager('\n'.join(classifier))


@show.command()
@click.argument('type', type=click.Choice(feature_list))
def feature(type):
    """
    Show all the features variation of the given  TYPE.
    """
    features = app.show_feature_variations(type)
    click.echo_via_pager('\n'.join(features))


# ############################ DELETE OPERATIONS ############################

@click1.group()
def delete():
    """
    Delete images, categories, classifier, features, etc.
    """
    pass


@delete.command()
@click.argument('category', type=click.Choice(categories_list), nargs=1)
@click.argument('images_name', required=True, nargs=-1)
@click.confirmation_option(prompt='Are you sure you want to delete the image/s?')
def image(category, images_name):
    """
    Delete the given IMAGES from the give CATEGORY set.
    """
    for img in images_name:
        try:
            app.delete_image_by_category(category, img)
        except FileExistsError:
            click.FileError("{0} does not exist".format(img))


@delete.command()
@click.argument('category', type=click.Choice(categories_list))
@click.confirmation_option(prompt='Are you sure you want to drop the category?')
def category(category):
    """
    Delete the given CATEGORY from the system and all the images attached to it.
    """
    app.delete_category(category)


@delete.command()
@click.argument('type', type=click.Choice(classifier_list), nargs=1)
@click.argument('parameters', required=True, nargs=-1)
@click.confirmation_option(prompt='Are you sure you want to drop the classifier?')
def classifier(type, parameters):
    """
    Delete the classifier with the given PARAMETERS among those of the given TYPE.
    """
    try:
        app.delete_classifier(type, parameters)
    except FileExistsError:
        click.FileError("The given classifier variation does not exist")


@delete.command()
@click.argument('type', type=click.Choice(feature_list), nargs=1)
@click.argument('parameters', required=True, nargs=-1)
@click.confirmation_option(prompt='Are you sure you want to drop the feature?')
def feature(type, parameters):
    """
    Delete the feature with the given PARAMETERS among those of the given TYPE.
    """
    try:
        app.delete_feature(type, parameters)
    except FileExistsError:
        click.FileError("The given feature variation does not exist")


# ############################ SET OPERATIONS ############################

@click1.group()
def set():
    """
    Change the default classifier' parameters and features' parameters.
    """
    pass


@set.command()
@click.argument('type', type=click.Choice(classifier_list), nargs=1)
@click.argument('parameters', required=True, nargs=-1)
def classifier(type, parameters):
    """
    Change the default PARAMETERS of the given classifier TYPE.
    """
    try:
        app.set_default_classifier(type, parameters)
    except FileExistsError:
        click.FileError("The given classifier variation does not exist")


@set.command()
@click.argument('type', type=click.Choice(feature_list), nargs=1)
@click.argument('parameters', required=True, nargs=-1)
def feature(type, parameters):
    """
    Change the default PARAMETERS of the given feature TYPE.
    """
    try:
        app.set_default_feature(type, parameters)
    except FileExistsError:
        click.FileError("The given feature variation does not exist")


# ############################ END ############################
# Add group to main click
cli = click.CommandCollection(sources=[click1],
                              help="This application allows to manage logos and logos's categories, "
                                   "to know the probability that a website is a webshop "
                                   "and to know the services offered by a webshop.")

if __name__ == '__main__':
    cli()
