import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from app import App


class Gui(QWidget):
    """
    Main GUI window for the application logorec.
    """

    def __init__(self):
        """
        Init the main window.
        """
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Init all the UI components of the window.

        :return: nothing
        """
        self.setWindowTitle('Logorec')
        self.setWindowIcon(QIcon('data/images/ico.png'))

        # TITLE
        title = QLabel()
        title.setText("LOGOREC")
        title.setStyleSheet("font-size:25pt; font-weight:bold; text-decoration:underline")
        title.setAlignment(Qt.AlignCenter)

        # TRAIN BUTTON
        trainButton = QPushButton("Train")
        trainButton.setStyleSheet("width: 300px; height:100px; font-size: 20pt")
        trainButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        trainButton.clicked.connect(self.trainButtonClicked)

        # DELETE BUTTON
        deleteButton = QPushButton("Delete")
        deleteButton.setStyleSheet("width: 300px; height:100px; font-size: 20pt")
        deleteButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        deleteButton.clicked.connect(self.deleteButtonClicked)

        # ADD BUTTON
        addButton = QPushButton("Add")
        addButton.setStyleSheet("width: 300px; height:100px; font-size: 20pt")
        addButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        addButton.clicked.connect(self.addButtonClicked)

        # GET BUTTON
        getButton = QPushButton("Get")
        getButton.setStyleSheet("width: 300px; height:100px; font-size: 20pt")
        getButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        getButton.clicked.connect(self.getButtonClicked)

        # MAIN LAYOUT
        vbox = QVBoxLayout()
        vbox.addWidget(title)
        vbox.addWidget(trainButton)
        vbox.addWidget(deleteButton)
        vbox.addWidget(addButton)
        vbox.addWidget(getButton)

        self.setLayout(vbox)

        self.center()

        self.show()

    def center(self):
        """
        Put the windows in the center of the screen.

        :return: nothing
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def trainButtonClicked(self):
        """
        Action called when the train button is clicked. It open the train window and close the current one.

        :return: nothing
        """
        self.close()
        self.window = Train()

    def addButtonClicked(self):
        """
        Action called when the add button is clicked. It open the add window and close the current one.

        :return: nothing
        """
        self.close()
        self.window = Add()

    def deleteButtonClicked(self):
        """
        Action called when the delete button is clicked. It open the delete window and close the current one.

        :return: nothing
        """
        self.close()
        self.window = Delete()

    def getButtonClicked(self):
        """
        Action called when the get button is clicked. It open the get window and close the current one.

        :return: nothing
        """
        self.close()
        self.window = Get()


class Add(QWidget):
    """
    GUI window to add new logos, classifier, feature to the logorec system.
    """

    def __init__(self):
        """
        Init the Add GUI window.
        """
        super().__init__()
        # Image List for the save process
        self.images = []
        # Load logorec app
        self.app = App()
        self.initUI()

    def initUI(self):
        """
        Init all the window components.

        :return: nothing
        """
        self.setWindowTitle('Logorec')
        self.setWindowIcon(QIcon('data/images/ico.png'))

        # GO BACK BUTTON
        backButton = QPushButton("Back")
        backButton.clicked.connect(self.back_button_clicked)

        # ERROR MESSAGE
        self.errorLabel = QLabel("")
        self.errorLabel.setStyleSheet("color:red")

        # TOP ELEMENTS
        hbox = QHBoxLayout()
        hbox.addWidget(backButton)
        hbox.addStretch()
        hbox.addWidget(self.errorLabel)

        # TITLE
        title = QLabel()
        title.setText("ADD")
        title.setStyleSheet("font-size:25pt; font-weight:bold; text-decoration:underline")
        title.setAlignment(Qt.AlignCenter)

        # ## CATEGORY ELEMENTS ##
        # TITLE
        categoryTitle = QLabel()
        categoryTitle.setText("Category")
        categoryTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # CATEGORY INPUT
        self.categoryEdit = QLineEdit()
        self.categoryEdit.textChanged.connect(self.disable_category_button)
        self.categoryEdit.setStyleSheet("width:200px; height:50px; font-size:10pt")

        # CATEGORY ADD BUTTON
        self.categoryButton = QPushButton("Add")
        self.categoryButton.setDisabled(True)
        self.categoryButton.clicked.connect(self.add_category_button_clicked)
        self.categoryButton.setStyleSheet("width:100px; height:50px; font-size:10pt")

        # CATEGORY INPUT AND ADD BUTTON IN THE SAME LINE
        categoryLine = QHBoxLayout()
        categoryLine.addWidget(self.categoryEdit)
        categoryLine.addWidget(self.categoryButton)

        # CATEGORY TITLE AND CATEGORY ELEMENTS
        category = QVBoxLayout()
        category.addWidget(categoryTitle)
        category.addLayout(categoryLine)

        # ## IMAGES ELEMENTS ##

        # IMAGES TITLE
        imagesTitle = QLabel()
        imagesTitle.setText("Images")
        imagesTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # IMAGES SELECT CATEGORY
        imagesSelectButton = QPushButton("Select")
        imagesSelectButton.clicked.connect(self.open_file_names_dialog)
        imagesSelectButton.setStyleSheet("width:50px; height:50px; font-size:10pt")

        # IMAGES ADD IMAGES TO CATEGORY
        self.imagesButton = QPushButton("Add")
        self.imagesButton.setDisabled(True)
        self.imagesButton.clicked.connect(self.save_images)
        self.imagesButton.setStyleSheet("width:50px; height:50px; font-size:10pt")

        # IMAGES CATEGORIES COMBO BOX
        self.imagesCombo = QComboBox(self)
        self.full_combo_images()
        self.imagesCombo.setStyleSheet("width:200px; height:50px; font-size:10pt")

        # IMAGES SELECT, ADD AND COMBO BOX IN THE SAME LINE
        imagesLine = QHBoxLayout()
        imagesLine.addWidget(self.imagesCombo)
        imagesLine.addWidget(imagesSelectButton)
        imagesLine.addWidget(self.imagesButton)

        # IMAGES TITLE AND IMAGES COMPONENTS
        images = QVBoxLayout()
        images.addWidget(imagesTitle)
        images.addLayout(imagesLine)

        # ## CATEGORY ELEMENTS ##

        # CLASSIFIER TITLE
        classifierTitle = QLabel()
        classifierTitle.setText("Classifier")
        classifierTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # CLASSIFIER ADD BUTTON
        self.classifierButton = QPushButton("Add")
        self.classifierButton.setDisabled(True)
        self.classifierButton.clicked.connect(self.add_classifier_button_clicked)
        self.classifierButton.setStyleSheet("width:100px; height:50px; font-size:10pt")

        # CLASSIFIERS COMBO BOX
        self.classifierCombo = QComboBox(self)
        self.full_combo_classifier()
        self.classifierCombo.setStyleSheet("width:300px; height:50px; font-size:10pt")

        # CLASSIFIER PARAMETERS INPUT
        self.classifierEdit = QLineEdit()
        self.classifierEdit.textChanged.connect(self.disable_classifier_button)
        self.classifierEdit.setStyleSheet("width:200px; height:50px; font-size:10pt")

        # CLASSIFIER ADD AND PARAMETERS INPUT IN THE SAME LINE
        classifierLine = QHBoxLayout()
        classifierLine.addWidget(self.classifierEdit)
        classifierLine.addWidget(self.classifierButton)

        # CLASSIFIER TITLE, ADD AND LINE
        classifier = QVBoxLayout()
        classifier.addWidget(classifierTitle)
        classifier.addWidget(self.classifierCombo)
        classifier.addLayout(classifierLine)

        # ## FEATURE ELEMENTS ##

        # FEATURE TITLE
        featureTitle = QLabel()
        featureTitle.setText("Feature")
        featureTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # FEATURE ADD BUTTON
        self.featureButton = QPushButton("Add")
        self.featureButton.setDisabled(True)
        self.featureButton.clicked.connect(self.add_feature_button_clicked)
        self.featureButton.setStyleSheet("width:100px; height:50px; font-size:10pt")

        # FEATURE COMBO BOX
        self.featureCombo = QComboBox(self)
        self.full_combo_feature()
        self.featureCombo.setStyleSheet("width:300px; height:50px; font-size:10pt")

        # FEATURE PARAMETERS INPUT
        self.featureEdit = QLineEdit()
        self.featureEdit.textChanged.connect(self.disable_feature_button)
        self.featureEdit.setStyleSheet("width:200px; height:50px; font-size:10pt")

        # FEATURE PARAMETERS INPUT AND BUTTON IN THE SAME LINE
        featureLine = QHBoxLayout()
        featureLine.addWidget(self.featureEdit)
        featureLine.addWidget(self.featureButton)

        # FEATURE TITLE, COMBO BOX AND LINE
        feature = QVBoxLayout()
        feature.addWidget(featureTitle)
        feature.addWidget(self.featureCombo)
        feature.addLayout(featureLine)

        # MAIN LAYOUT

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(title)
        vbox.addLayout(category)
        vbox.addLayout(images)
        vbox.addLayout(classifier)
        vbox.addLayout(feature)

        self.setLayout(vbox)

        self.show()

    def back_button_clicked(self):
        """
        Close current window and go back to the previous one.

        :return: nothing
        """
        self.close()
        self.window = Gui()

    def update(self):
        """
        Update add components after a new addition (update combo boxes)

        :return: nothing
        """
        self.app = App()
        self.imagesCombo.clear()
        self.full_combo_images()
        self.imagesCombo.update()

    def add_category_button_clicked(self):
        """
        If the new category is not already present add it to the system otherwise, display an error.

        :return: nothing
        """
        # When a new action is called reset error message
        self.errorLabel.setText("")
        value = self.categoryEdit.text().upper()
        if value not in self.app.get_categories():
            self.app.add_category(value)
            self.update()
        else:
            self.errorLabel.setText("The given category already exists")
        # Reset parameters and button
        self.categoryEdit.setText("")
        self.categoryButton.setDisabled(True)

    def disable_category_button(self):
        """
        Disable the category add button if there is not text on the category input field otherwise, enable it.

        :return: nothing
        """
        if len(self.categoryEdit.text()) > 0:
            self.categoryButton.setDisabled(False)
        else:
            self.categoryButton.setDisabled(True)

    def disable_classifier_button(self):
        """
        Disable the classifier add button if there is not text on the category input field otherwise, enable it.

        :return: nothing
        """
        if len(self.classifierEdit.text()) > 0:
            self.classifierButton.setDisabled(False)
        else:
            self.classifierButton.setDisabled(True)

    def disable_feature_button(self):
        """
        Disable the feature add button if there is not text on the category input field otherwise, enable it.

        :return: nothing
        """
        if len(self.featureEdit.text()) > 0:
            self.featureButton.setDisabled(False)
        else:
            self.featureButton.setDisabled(True)

    def full_combo_images(self):
        """
        Fill up the categories combo box.

        :return: nothing
        """
        categories = self.app.get_categories()
        for cat in categories:
            self.imagesCombo.addItem(cat)

    def open_file_names_dialog(self):
        """
        Open a file dialogs to add all the new images.

        :return: nothing
        """
        # Reset images each time they are added
        self.images = []
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "All Files (*);;Python Files (*.py)", options=options)
        # If there are files
        if files:
            for f in files:
                self.images.append(f)
            # If there is at least one logo category enable add button
            if len(self.imagesCombo.currentText()) > 0:
                self.imagesButton.setDisabled(False)

    def save_images(self):
        """
        Save the added images to the system.

        :return: nothing
        """
        # When a new action is called reset error message
        self.errorLabel.setText("")
        for img in self.images:
            self.app.add_image(img, self.imagesCombo.currentText())
        self.imagesButton.setDisabled(True)
        self.images = []

    def full_combo_classifier(self):
        """
        Fill up the classifier combo box.

        :return: nothing
        """
        classifiers = self.app.get_classifiers()
        for cls in classifiers:
            self.classifierCombo.addItem(cls)

    def add_classifier_button_clicked(self):
        """
        Add the new classifier variation to the system.

        :return: nothing
        """
        # When a new action is called reset error message
        self.errorLabel.setText("")
        try:
            self.app.add_classifier(self.classifierCombo.currentText(), self.classifierEdit.text().split(" "))
        except AttributeError:
            self.errorLabel.setText("Parameters do not correspond with the classifier implementation")
        except FileExistsError:
            self.errorLabel.setText("The classifier variation already exists")

        self.classifierEdit.setText("")
        self.classifierButton.setDisabled(True)

    def full_combo_feature(self):
        """
        Fill up feature combo box.

        :return: nothing
        """
        features = self.app.get_features()
        for f in features:
            self.featureCombo.addItem(f)

    def add_feature_button_clicked(self):
        """
        Add the new feature variation to the system.

        :return: nothing
        """
        # When a new action is called reset error message
        self.errorLabel.setText("")
        try:
            self.app.add_feature(self.featureCombo.currentText(), self.featureEdit.text().split(" "))
        except AttributeError:
            self.errorLabel.setText("Parameters do not correspond with the feature implementation")
        except FileExistsError:
            self.errorLabel.setText("The feature variation already exists")
        self.featureEdit.setText("")
        self.featureButton.setDisabled(True)


class Delete(QWidget):
    """
    GUI window to delete logos, classifier, feature from the logorec system.
    """

    def __init__(self):
        """
        Init the Delete GUI window.
        """
        super().__init__()
        self.app = App()
        self.initUI()

    def initUI(self):
        """
        Init all the window components.

        :return:nothing
        """
        self.setWindowTitle('Logorec')
        self.setWindowIcon(QIcon('data/images/ico.png'))

        # BACK BUTTN
        backButton = QPushButton("Back")
        backButton.clicked.connect(self.back_button_clicked)

        # ERROR MESSAGE
        self.errorLabel = QLabel("")
        self.errorLabel.setStyleSheet("color:red")

        # TOP ELEMENTS
        hbox = QHBoxLayout()
        hbox.addWidget(backButton)
        hbox.addStretch()
        hbox.addWidget(self.errorLabel)

        # TITLE
        title = QLabel()
        title.setText("DELETE")
        title.setStyleSheet("font-size:25pt; font-weight:bold; text-decoration:underline")
        title.setAlignment(Qt.AlignCenter)

        # ## CATEGORY ##

        # CATEGORY TITLE
        categoryTitle = QLabel()
        categoryTitle.setText("Category")
        categoryTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # CATEGORY DELETE BUTTON
        self.categoryButton = QPushButton("Delete")
        self.categoryButton.clicked.connect(self.delete_category_button_clicked)
        self.categoryButton.setStyleSheet("width:50px; height:50px; font-size:10pt")

        # CATEGORY COMBO BOX
        self.categoryCombo = QComboBox()
        self.categoryCombo.setStyleSheet("width:250px; height:50px; font-size:10pt")
        self.full_combo_categories()

        # CATEGORY DELETE BUTTON AND COMBO BOX ON THE SAME LINE
        categoryLine = QHBoxLayout()
        categoryLine.addWidget(self.categoryCombo)
        categoryLine.addWidget(self.categoryButton)

        # CATEGORY TITLE AND LINE
        category = QVBoxLayout()
        category.addWidget(categoryTitle)
        category.addLayout(categoryLine)

        # ## IMGES ##

        # IMAGES TITLE
        imagesTitle = QLabel()
        imagesTitle.setText("Images")
        imagesTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # IMAGES SEE BUTTON
        self.imagesSeeButton = QPushButton("See")
        self.imagesSeeButton.clicked.connect(self.open_file_names_dialog)
        self.imagesSeeButton.setStyleSheet("width:50px; height:50px; font-size:10pt")

        # IMAGES CATEGORIES COMBO BOX
        self.imagesCombo = QComboBox(self)
        self.full_combo_images()
        self.imagesCombo.setStyleSheet("width:250px; height:50px; font-size:10pt")

        # IMAGES COMBO BOX AND SEE BUTTON SAME LINE
        imagesLine = QHBoxLayout()
        imagesLine.addWidget(self.imagesCombo)
        imagesLine.addWidget(self.imagesSeeButton)

        # IMAGES TITLE AND LINE
        images = QVBoxLayout()
        images.addWidget(imagesTitle)
        images.addLayout(imagesLine)

        # ## CLASSIFIER ##

        # CLASSIFIER TITLE
        classifierTitle = QLabel()
        classifierTitle.setText("Classifier")
        classifierTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # CLASSIFIER DELETE BUTTON
        self.classifierButton = QPushButton("Delete")
        self.classifierButton.clicked.connect(self.delete_classifier_button_clicked)
        self.classifierButton.setStyleSheet("width:300px; height:50px; font-size:10pt")

        # CLASSIFIER COMBO BOX
        self.classifierCombo = QComboBox(self)
        self.full_combo_classifier()
        self.classifierCombo.currentTextChanged.connect(self.full_combo_classifier_variation)
        self.classifierCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIER VARIATION COMBO BOX
        self.classifierVariationCombo = QComboBox()
        self.full_combo_classifier_variation()
        self.classifierVariationCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIER COMBO BOXES IN THE SAME LINE
        classifierLine = QHBoxLayout()
        classifierLine.addWidget(self.classifierCombo)
        classifierLine.addWidget(self.classifierVariationCombo)

        # CLASSIFIER TITLE, LINE AND DELETE BUTTON
        classifier = QVBoxLayout()
        classifier.addWidget(classifierTitle)
        classifier.addLayout(classifierLine)
        classifier.addWidget(self.classifierButton)

        # ## FEATURE ##

        # FEATURE TITLE
        featureTitle = QLabel()
        featureTitle.setText("Feature")
        featureTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # FEATURE DELETE BUTTON
        self.featureDeleteButton = QPushButton("Delete")
        self.featureDeleteButton.clicked.connect(self.delete_feature_button_clicked)
        self.featureDeleteButton.setStyleSheet("width:300px; height:50px; font-size:10pt")

        # FEATURE COMBO BOX
        self.featureCombo = QComboBox(self)
        self.full_combo_feature()
        self.featureCombo.currentTextChanged.connect(self.full_combo_feature_variation)
        self.featureCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # FEATURE VARIATION COMBO BOX
        self.featureVariationCombo = QComboBox()
        self.full_combo_feature_variation()
        self.featureVariationCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # FEATURE COMBO BOXES IN THE SAME LINE
        featureLine = QHBoxLayout()
        featureLine.addWidget(self.featureCombo)
        featureLine.addWidget(self.featureVariationCombo)

        # FEATURE TITLE, LINE AND DELETE BUTTON
        feature = QVBoxLayout()
        feature.addWidget(featureTitle)
        feature.addLayout(featureLine)
        feature.addWidget(self.featureDeleteButton)

        # MAIN LAYOUT

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(title)
        vbox.addLayout(category)
        vbox.addLayout(images)
        vbox.addLayout(classifier)
        vbox.addLayout(feature)

        self.setLayout(vbox)

        self.show()

    def back_button_clicked(self):
        """
        Close current window and go back to the previous one.

        :return: nothing
        """
        self.close()
        self.window = Gui()

    def update(self):
        """
        Update all the components after that a delation has occurred.

        :return: nothing
        """
        self.app = App()

        # CATEGORIES
        self.categoryCombo.clear()
        self.full_combo_categories()
        self.categoryCombo.update()

        # IMAGE CATEGORIES
        self.imagesCombo.clear()
        self.full_combo_images()
        self.imagesCombo.update()

        # FEATURES
        self.featureCombo.clear()
        self.full_combo_feature()
        self.featureCombo.update()

        # FEATURE VARIATIONS
        self.featureVariationCombo.clear()
        self.full_combo_feature_variation()
        self.featureVariationCombo.update()

        # CLASSIFIERS
        self.classifierCombo.clear()
        self.full_combo_classifier()
        self.classifierCombo.update()

        # CLASSIFIER VARIATIONS
        self.classifierVariationCombo.clear()
        self.full_combo_classifier_variation()
        self.classifierVariationCombo.update()

    def delete_category_button_clicked(self):
        """
        Display a confirmation message before deleting the choosen category.

        :return: nothing
        """
        buttonReply = QMessageBox.question(self, 'Confirm message', "Are you sure to delete the category?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            self.app.delete_category(self.categoryCombo.currentText())
            self.update()

    def full_combo_categories(self):
        """
        Fill up the categories combo box.

        :return: nothing
        """
        categories = self.app.get_categories()
        if not categories:
            # If there are not categories, disable the delete button
            self.categoryButton.setDisabled(True)
        for cat in categories:
            self.categoryCombo.addItem(cat)

    def full_combo_images(self):
        """
        Fill up the categories for the images deletion.

        :return: nothing
        """
        categories = self.app.get_categories()
        if not categories:
            # If there are not categories, disable the see images button
            self.imagesSeeButton.setDisabled(True)
        for cat in categories:
            self.imagesCombo.addItem(cat)

    def open_file_names_dialog(self):
        """
        Open the delete imges for the choosen category.

        :return: nothing
        """
        self.close()
        self.window = DeleteImage(self.imagesCombo.currentText())

    def full_combo_classifier(self):
        """
        Fill up the classifier combo box.

        :return: nothing
        """
        classifiers = self.app.get_classifiers()
        if not classifiers:
            # If there are not classifiers, disable the delete button
            self.classifierButton.setDisabled(True)
        for cls in classifiers:
            self.classifierCombo.addItem(cls)

    def full_combo_classifier_variation(self):
        """
        Fill up the classifier variations combo box.

        :return: nothing
        """
        # If the classifier combo box is empty (no classifier) do nothing
        if self.classifierCombo.currentText() != "":
            val = self.classifierCombo.currentText()
            # Get selected classifier variation
            variations = self.app.show_classifier_variations(val)
            if not variations:
                # If there are not variations disable the delete button
                self.classifierButton.setDisabled(True)
            for v in variations:
                # If there is the default keyword delete it
                if v.split(" ")[0] == 'default':
                    self.classifierVariationCombo.addItem(" ".join(v.split(" ")[1:]))
                else:
                    self.classifierVariationCombo.addItem(v)

    def delete_classifier_button_clicked(self):
        """
        Display confirmation message before deleting the selected  classifier variation.

        :return: nothing
        """
        buttonReply = QMessageBox.question(self, 'Confirm message', "Are you sure to delete the classifier variation?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            val = self.classifierCombo.currentText()
            var = self.classifierVariationCombo.currentText().split(" ")
            self.app.delete_classifier(val, var)
            self.update()

    def full_combo_feature(self):
        """
        Fill up the feature combo box.

        :return: nothing
        """
        features = self.app.get_features()
        if not features:
            self.featureDeleteButton.setDisabled(True)
        for f in features:
            self.featureCombo.addItem(f)

    def full_combo_feature_variation(self):
        """
        Fill up the feature variations combo box.

        :return: nothing
        """
        # If the classifier combo box is empty (no feature) do nothing
        if self.featureCombo.currentText() != "":
            val = self.featureCombo.currentText()
            # Get selected feature variation
            variations = self.app.show_feature_variations(val)
            if not variations:
                self.featureDeleteButton.setDisabled(True)
            for v in variations:
                # If there is the default keyword delete it
                if v.split(" ")[0] == 'default':
                    self.featureVariationCombo.addItem(" ".join(v.split(" ")[1:]))
                else:
                    self.featureVariationCombo.addItem(v)

    def delete_feature_button_clicked(self):
        """
        Display confirmation dialog before deleting the selected feature variation.

        :return: nothing
        """
        buttonReply = QMessageBox.question(self, 'Confirm message', "Are you sure to delete the feature variation?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            val = self.featureCombo.currentText()
            var = self.featureVariationCombo.currentText().split(" ")
            self.app.delete_feature(val, var)
            self.update()


class DeleteImage(QWidget):
    """
    Window to delete images of the given category.
    """

    def __init__(self, category):
        """
        Init the DeleteImage window.

        :param category: Image category in which images must be deleted
        """
        super().__init__()
        self.category = category
        self.app = App()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Logorec')
        self.setWindowIcon(QIcon('data/images/ico.png'))

        # BACK BUTTON
        backButton = QPushButton("Back")
        backButton.clicked.connect(self.back_button_clicked)

        # ERROR MESSAGE
        self.errorLabel = QLabel("")
        self.errorLabel.setStyleSheet("color:red")

        # TOP ELEMENTS
        hbox = QHBoxLayout()
        hbox.addWidget(backButton)
        hbox.addStretch()
        hbox.addWidget(self.errorLabel)

        # TITLE
        title = QLabel()
        title.setText("DELETE IMAGES")
        title.setStyleSheet("font-size:25pt; font-weight:bold; text-decoration:underline")
        title.setAlignment(Qt.AlignCenter)

        # ## IMAGES ##

        # IMAGES DELETE BUTTON
        self.imagesButton = QPushButton("Delete")
        self.imagesButton.clicked.connect(self.delete_category_button_clicked)
        self.imagesButton.setStyleSheet("width:300px; height:50px; font-size:10pt")
        self.imagesButton.setDisabled(True)

        # IMAGES LIST
        self.imagesList = QListWidget()

        # fill up images list
        for img in self.app.show_images_by_category(self.category):
            part = img.split("\\")
            self.imagesList.addItem(part[len(part) - 1])

        self.imagesList.itemSelectionChanged.connect(self.selection_changed)
        self.imagesList.setSelectionMode(QAbstractItemView.MultiSelection)

        # MAIN LAYOUT
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(title)
        vbox.addWidget(self.imagesList)
        vbox.addWidget(self.imagesButton)

        self.setLayout(vbox)

        self.show()

    def back_button_clicked(self):
        """
        Close current window and go back to the previous one.

        :return: nothing
        """
        self.close()
        self.window = Delete()

    def selection_changed(self):
        """
        Enable or disable the image delete button if elements are selected or not.

        :return: nothing
        """
        if self.imagesList.selectedItems():
            self.imagesButton.setDisabled(False)
        else:
            self.imagesButton.setDisabled(True)

    def delete_category_button_clicked(self):
        """
        Display a confirmation pup up before deleting the selected images. Once the delation is completed go back to
        the previous window.

        :return:nothing
        """
        buttonReply = QMessageBox.question(self, 'Confirm message', "Are you sure to delete the selected images?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            for img in self.imagesList.selectedItems():
                self.app.delete_image_by_category(self.category, img.text())
            self.back_button_clicked()


class Train(QWidget):
    """
    Window to train different classifier and feature.
    """

    def __init__(self):
        """
        Init the train window
        """
        super().__init__()
        self.images = []
        self.app = App()
        self.initUI()

    def initUI(self):
        """
        Init all the window components.

        :return: nothing
        """
        self.setWindowTitle('Logorec')
        self.setWindowIcon(QIcon('data/images/ico.png'))

        # GO BACK BUTTON
        backButton = QPushButton("Back")
        backButton.clicked.connect(self.back_button_clicked)

        # ERROR MESSAGE
        self.errorLabel = QLabel("")
        self.errorLabel.setStyleSheet("color:red")

        # TOP ELEMENTS
        hbox = QHBoxLayout()
        hbox.addWidget(backButton)
        hbox.addStretch()
        hbox.addWidget(self.errorLabel)

        # TITLE
        title = QLabel()
        title.setText("TRAIN")
        title.setStyleSheet("font-size:25pt; font-weight:bold; text-decoration:underline")
        title.setAlignment(Qt.AlignCenter)

        # ## CLASSIFIER ##

        # CLASSIFIER TITLE
        classifierTitle = QLabel()
        classifierTitle.setText("Classifier")
        classifierTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # CLASSIFIER TRAIN BUTTON
        self.classifierButton = QPushButton("Train")
        self.classifierButton.clicked.connect(self.train_classifier_button_clicked)
        self.classifierButton.setStyleSheet("width:300px; height:50px; font-size:10pt")

        # CLASSIFIER FEATURES COMBO BOX
        self.featureClassifierCombo = QComboBox(self)
        self.full_combo_feature_classifier()
        self.featureClassifierCombo.currentTextChanged.connect(self.full_combo_feature_classifier_variation)
        self.featureClassifierCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIER FEATURE VARIATIONS COMBO BOX
        self.featureClassifierVariationCombo = QComboBox()
        self.full_combo_feature_classifier_variation()
        self.featureClassifierVariationCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIERS COMBO BOX
        self.classifierCombo = QComboBox(self)
        self.full_combo_classifier()
        self.classifierCombo.currentTextChanged.connect(self.full_combo_classifier_variation)
        self.classifierCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIER VARIATIONS COMBO BOX
        self.classifierVariationCombo = QComboBox()
        self.full_combo_classifier_variation()
        self.classifierVariationCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIER FEATURES AND FEATURE VARIATIONS IN THE SAME LINE
        classifierFeatureLine = QHBoxLayout()
        classifierFeatureLine.addWidget(self.featureClassifierCombo)
        classifierFeatureLine.addWidget(self.featureClassifierVariationCombo)

        # CLASSIFIERS AND VARIATIONS IN THE SAME LINE
        classifierLine = QHBoxLayout()
        classifierLine.addWidget(self.classifierCombo)
        classifierLine.addWidget(self.classifierVariationCombo)

        # CLASSIFIER TITLE, FEATURES LINE, CLASSIFIERS LINE AND TRAIN BUTTON
        classifier = QVBoxLayout()
        classifier.addWidget(classifierTitle)
        classifier.addLayout(classifierFeatureLine)
        classifier.addLayout(classifierLine)
        classifier.addWidget(self.classifierButton)

        # ## FEATURE ##

        # FEATURE TITLE
        featureTitle = QLabel()
        featureTitle.setText("Feature")
        featureTitle.setStyleSheet("font-size:20pt; text-decoration:underline")

        # FEATURE TRAIN BUTTON
        self.featureButton = QPushButton("Train")
        self.featureButton.clicked.connect(self.train_feature_button_clicked)
        self.featureButton.setStyleSheet("width:300px; height:50px; font-size:10pt")

        # FEATURES COMBO BOX
        self.featureCombo = QComboBox(self)
        self.full_combo_feature()
        self.featureCombo.currentTextChanged.connect(self.full_combo_feature_variation)
        self.featureCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # FEATURE VARIATIONS COMBO BOX
        self.featureVariationCombo = QComboBox()
        self.full_combo_feature_variation()
        self.featureVariationCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # FEATURES AND VARIATIONS IN THE SAME LINE
        featureLine = QHBoxLayout()
        featureLine.addWidget(self.featureCombo)
        featureLine.addWidget(self.featureVariationCombo)

        # FEATURE TITLE, LINE AND TRAIN BUTTON
        feature = QVBoxLayout()
        feature.addWidget(featureTitle)
        feature.addLayout(featureLine)
        feature.addWidget(self.featureButton)

        # MAIN LAYOUT

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(title)
        vbox.addLayout(feature)
        vbox.addLayout(classifier)

        self.setLayout(vbox)

        self.show()

    def back_button_clicked(self):
        """
        Close current window and go back to the previous one.

        :return: nothing
        """
        self.close()
        self.window = Gui()

    def full_combo_classifier(self):
        """
        Fill up the classifier combo box.

        :return: nothing
        """
        classifiers = self.app.get_classifiers()
        if not classifiers:
            self.classifierButton.setDisabled(True)
        for cls in classifiers:
            self.classifierCombo.addItem(cls)

    def full_combo_classifier_variation(self):
        """
        Fill up the classifier variation combo box.

        :return: nothing
        """
        self.errorLabel.setText("")
        # If there are not classifier available do nothing
        if self.classifierCombo.currentText() != "":
            val = self.classifierCombo.currentText()
            # Load variation of the selected classifier
            variations = self.app.show_classifier_variations(val)
            if not variations:
                self.classifierButton.setDisabled(True)
            for v in variations:
                # Delete default keyword if it is present
                if v.split(" ")[0] == 'default':
                    self.classifierVariationCombo.addItem(" ".join(v.split(" ")[1:]))
                else:
                    self.classifierVariationCombo.addItem(v)

    def full_combo_feature_classifier(self):
        """
        Fill up the classifier feature combo box.

        :return: nothing
        """
        features = self.app.get_features()
        if not features:
            self.classifierButton.setDisabled(True)
        for f in features:
            self.featureClassifierCombo.addItem(f)

    def full_combo_feature_classifier_variation(self):
        """
        Fill up the classifier feature variation combo box.

        :return: nothing
        """
        # If the there are not feature available do nothing
        if self.featureClassifierCombo.currentText() != "":
            val = self.featureClassifierCombo.currentText()
            # Load variations
            variations = self.app.show_feature_variations(val)
            if not variations:
                self.classifierButton.setDisabled(True)
            for v in variations:
                # if there is the default keyword delete it
                if v.split(" ")[0] == 'default':
                    self.featureClassifierVariationCombo.addItem(" ".join(v.split(" ")[1:]))
                else:
                    self.featureClassifierVariationCombo.addItem(v)


    def train_classifier_button_clicked(self):
        """
        Display confirmation message before training the selected classifier with the selected feature. If problems
        occur a error message is displayed.

        :return: nothing
        """
        self.errorLabel.setText("")
        buttonReply = QMessageBox.question(self, 'Confirm message', "Are you sure to train the classifier variation?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            classifier = self.classifierCombo.currentText()
            classifier_variation = self.classifierVariationCombo.currentText().split(" ")
            feature = self.featureClassifierCombo.currentText()
            feature_variation = self.featureClassifierVariationCombo.currentText().split(" ")
            try:
                self.app.set_default_feature(feature, feature_variation)
                self.app.set_default_classifier(classifier, classifier_variation)
                self.app.train_classifier(classifier, feature)
            except ModuleNotFoundError:
                self.errorLabel.setText("Feature is not trained")

    def full_combo_feature(self):
        """
        Fill up the features combo box.

        :return: nothing
        """
        features = self.app.get_features()
        if not features:
            self.featureButton.setDisabled(True)
        for f in features:
            self.featureCombo.addItem(f)

    def full_combo_feature_variation(self):
        """
        Fill up the feature variations combo box.

        :return: nothing
        """
        # if there are not available feature do nothing
        if self.featureCombo.currentText() != "":
            val = self.featureCombo.currentText()
            # load variations
            variations = self.app.show_feature_variations(val)
            if not variations:
                self.featureButton.setDisabled(True)
            for v in variations:
                # if there is the default keyword delete it
                if v.split(" ")[0] == 'default':
                    self.featureVariationCombo.addItem(" ".join(v.split(" ")[1:]))
                else:
                    self.featureVariationCombo.addItem(v)

    def train_feature_button_clicked(self):
        """
        Display confirmation message before training the selected feature variation. If errors occure an error message
        is displayed.
        :return:
        """
        buttonReply = QMessageBox.question(self, 'Confirm message', "Are you sure to train the feature variation?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            feature = self.featureCombo.currentText()
            variations = self.featureVariationCombo.currentText().split(" ")
            self.app.set_default_feature(feature, variations)
            self.app.train_feature(feature)


class Get(QWidget):
    """
    Window to retrieve probabilities and services of a given website
    """

    def __init__(self):
        """
        Init the get window.
        """
        super().__init__()
        self.app = App()
        self.initUI()

    def initUI(self):
        """
        Init all the components of the window

        :return: nothing
        """
        self.setWindowTitle('Logorec')
        self.setWindowIcon(QIcon('data/images/ico.png'))

        # GO BACK BUTTON
        backButton = QPushButton("Back")
        backButton.clicked.connect(self.back_button_clicked)

        # ERROR MESSAGE
        self.errorLabel = QLabel("")
        self.errorLabel.setStyleSheet("color:red")

        # TOP ELEMENTS
        topElements = QHBoxLayout()
        topElements.addWidget(backButton)
        topElements.addStretch()
        topElements.addWidget(self.errorLabel)

        # TITLE
        title = QLabel()
        title.setText("GET")
        title.setStyleSheet("font-size:25pt; font-weight:bold; text-decoration:underline")
        title.setAlignment(Qt.AlignCenter)

        # ## WEBSITE ##

        # ADD WEBSITE INPUT
        self.addWebsite = QLineEdit()
        self.addWebsite.setStyleSheet("width:250px; height:50px; font-size:10pt")
        self.addWebsite.textChanged.connect(self.add_website_changed)
        self.addWebsite.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ADD WEBSITE BUTTOn
        self.websiteButton = QPushButton("Add")
        self.websiteButton.clicked.connect(self.add_website_clicked)
        self.websiteButton.setStyleSheet("width:50px; height:50px; font-size:10pt")
        self.websiteButton.setDisabled(True)

        # ADD WEBSITE BUTTON AND INPUT IN THE SAME LINE
        website_layout = QHBoxLayout()
        website_layout.addWidget(self.addWebsite)
        website_layout.addWidget(self.websiteButton)

        # WEBSITE LIST
        self.websiteList = QListWidget()
        self.websiteList.itemSelectionChanged.connect(self.selectionChanged)

        # REMOVE WEBSITE BUTTON
        self.deleteWebsiteButton = QPushButton("Remove")
        self.deleteWebsiteButton.clicked.connect(self.delete_website_clicked)
        self.deleteWebsiteButton.setStyleSheet("width:50px; height:50px; font-size:10pt")
        self.deleteWebsiteButton.setDisabled(True)

        # ## CLASSIFIER ##

        # CLASSIFIER FEATURES COMBO BOX
        self.featureClassifierCombo = QComboBox(self)
        self.full_combo_feature_classifier()
        self.featureClassifierCombo.currentTextChanged.connect(self.full_combo_feature_classifier_variation)
        self.featureClassifierCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIER FEATURE VARIATIONS COMBO BOX
        self.featureClassifierVariationCombo = QComboBox()
        self.full_combo_feature_classifier_variation()
        self.featureClassifierVariationCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIERS COMBO BOX
        self.classifierCombo = QComboBox(self)
        self.full_combo_classifier()
        self.classifierCombo.currentTextChanged.connect(self.full_combo_classifier_variation)
        self.classifierCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIER VARIATIONS COMBO BOX
        self.classifierVariationCombo = QComboBox()
        self.full_combo_classifier_variation()
        self.classifierVariationCombo.setStyleSheet("width:150px; height:50px; font-size:10pt")

        # CLASSIFIER FEATURE COMBO BOXES
        classifierFeatureLine = QHBoxLayout()
        classifierFeatureLine.addWidget(self.featureClassifierCombo)
        classifierFeatureLine.addWidget(self.featureClassifierVariationCombo)

        # CLASSIFIER COMBO BOXES
        classifierLine = QHBoxLayout()
        classifierLine.addWidget(self.classifierCombo)
        classifierLine.addWidget(self.classifierVariationCombo)

        # ## BUTTONS ##

        # PROBABILITY BUTTON
        self.probabilityButton = QPushButton("Probability")
        self.probabilityButton.clicked.connect(self.probability_clicked)
        self.probabilityButton.setStyleSheet("width:150px; height:50px; font-size:10pt")
        self.probabilityButton.setDisabled(True)

        # SERVICES BUTTON
        self.servicesButton = QPushButton("Services")
        self.servicesButton.clicked.connect(self.services_clicked)
        self.servicesButton.setStyleSheet("width:150px; height:50px; font-size:10pt")
        self.servicesButton.setDisabled(True)

        # PROBABILITY AND SERVICES PARAMETERS INPUT
        self.extractionParameters = QLineEdit()
        self.extractionParameters.setStyleSheet("width:250px; height:50px; font-size:10pt")
        self.extractionParameters.textChanged.connect(self.extraction_parameters)

        # SERVICES AND PROBABILITY BUTTONS IN THE SAME LINE
        actions = QHBoxLayout()
        actions.addWidget(self.probabilityButton)
        actions.addWidget(self.servicesButton)

        # ## RESULTS ##

        # RESULT OUTPUT
        self.results = QLabel()
        self.results.setStyleSheet("width:300px; height:50px; font-size:10pt")

        # MAIN LAYOUT

        vbox = QVBoxLayout()
        vbox.addLayout(topElements)
        vbox.addWidget(title)
        vbox.addLayout(website_layout)
        vbox.addWidget(self.websiteList)
        vbox.addWidget(self.deleteWebsiteButton)
        vbox.addLayout(classifierFeatureLine)
        vbox.addLayout(classifierLine)
        vbox.addWidget(self.extractionParameters)
        vbox.addLayout(actions)
        vbox.addWidget(self.results)

        self.setLayout(vbox)

        self.show()

    def back_button_clicked(self):
        """
        Close current windows and go back to the previous one.

        :return: nothing
        """
        self.close()
        self.window = Gui()

    def probability_clicked(self):
        """
        Compute the probability that all the given website are webshop and print the results. If errors occure a error
        message is displayed.

        :return: nothing
        """
        try:
            # Reset error message if a new action is executed
            self.errorLabel.setText("")
            # Reset results
            self.results.setText("")
            # Set feature and classifier variation
            self.app.set_default_feature(self.featureClassifierCombo.currentText(),
                                         self.featureClassifierVariationCombo.currentText().split(" "))
            self.app.set_default_classifier(self.classifierCombo.currentText(),
                                            self.classifierVariationCombo.currentText().split(" "))
            # Loop over all websites
            for index in range(self.websiteList.count()):
                w = str(self.websiteList.item(index).text())
                prob = self.app.get_probability(w, self.extractionParameters.text().split(" "), self.classifierCombo.currentText(),
                                                self.featureClassifierCombo.currentText())
                # Print results
                self.results.setText(
                    self.results.text() + "\n {0} is a web shop with a probability of {1:2.2f} %".format(w, prob))
            # Disable buttons and reset values
            self.probabilityButton.setDisabled(True)
            self.servicesButton.setDisabled(True)
            self.extractionParameters.setText("")
            self.websiteList.clear()
        except AttributeError:
            self.errorLabel.setText("Websites do not represent implementation")
        except ValueError:
            self.errorLabel.setText("Parameters do not represent implementation")
        except ModuleNotFoundError:
            self.errorLabel.setText("Classifier is not trained")
        except FileNotFoundError:
            self.errorLabel.setText("Classifier is not trained correctly")
        finally:
            self.probabilityButton.setDisabled(True)
            self.servicesButton.setDisabled(True)

    def services_clicked(self):
        """
        Compute the probability of all the services present in the given websites. If errors occur an error message is
        displayed.

        :return: nothing
        """
        try:
            # Reset error message if a new action is executed
            self.errorLabel.setText("")
            # Reset results
            self.results.setText("")
            # Set  feature and classifier variation
            self.app.set_default_feature(self.featureClassifierCombo.currentText(),
                                         self.featureClassifierVariationCombo.currentText().split(" "))
            self.app.set_default_classifier(self.classifierCombo.currentText(),
                                            self.classifierVariationCombo.currentText().split(" "))
            # Loop over websites
            for index in range(self.websiteList.count()):
                w = str(self.websiteList.item(index).text())
                probs = self.app.get_services(w, self.extractionParameters.text().split(" "), self.classifierCombo.currentText(),
                                              self.featureClassifierCombo.currentText())
                # print results
                for cat, prob in zip(self.app.get_categories(), probs):
                    self.results.setText(
                        self.results.text() + "\n {0} is a offered by {1} with a probability of {2:2.2f} %".format(cat, w,
                                                                                                              prob))
                # Disable buttons and reset values
                self.probabilityButton.setDisabled(True)
                self.servicesButton.setDisabled(True)
                self.extractionParameters.setText("")
                self.websiteList.clear()
        except AttributeError:
            self.errorLabel.setText("Websites do not represent implementation")
        except ValueError:
            self.errorLabel.setText("Parameters does not represent implementation")
        except ModuleNotFoundError:
            self.errorLabel.setText("Classifier is not trained")
        except FileNotFoundError:
            self.errorLabel.setText("Classifier is not trained correctly")
        finally:
            self.probabilityButton.setDisabled(True)
            self.servicesButton.setDisabled(True)

    def extraction_parameters(self):
        """
        If the extraction parameters and the other needed value are available the probability and services button are
        enable. Otherwise, they are disabled.

        :return: nothing
        """
        # Reset error message if a new action is executed
        self.errorLabel.setText("")
        if len(self.extractionParameters.text()) > 0 and self.classifierVariationCombo.currentText() != "" \
                and self.featureClassifierVariationCombo.currentText() != "" and self.websiteList.count() != 0:
            self.errorLabel.setText("")
            self.probabilityButton.setDisabled(False)
            self.servicesButton.setDisabled(False)
        else:
            self.probabilityButton.setDisabled(True)
            self.servicesButton.setDisabled(True)

    def selectionChanged(self):
        """
        If there are selected websites the delete website button is enabled.

        :return: nothing
        """
        if self.websiteList.selectedItems():
            self.deleteWebsiteButton.setDisabled(False)
        else:
            self.deleteWebsiteButton.setDisabled(True)

    def delete_website_clicked(self):
        """
        Delete the selected websites.

        :return: nothing
        """
        # Reset error message if a new action is executed
        self.errorLabel.setText("")
        item = self.websiteList.selectedItems()[0]
        self.websiteList.takeItem(self.websiteList.row(item))
        self.deleteWebsiteButton.setDisabled(True)
        # If there are no websites the services and probability buttons are disabled
        if self.websiteList.count() == 0:
            self.probabilityButton.setDisabled(True)
            self.servicesButton.setDisabled(True)

    def add_website_clicked(self):
        """
        Add website to the websites list. If all the needed parameters are available the probability and services
        buttons are enabled.

        :return: nothing
        """
        # Reset error message if a new action is executed
        self.errorLabel.setText("")
        self.websiteList.addItem(self.addWebsite.text())
        self.addWebsite.setText("")
        self.websiteButton.setDisabled(True)
        # Enable probability and services buttons if all the needed parameters are available.
        if len(self.extractionParameters.text()) > 0 and self.classifierVariationCombo.currentText() != "" \
                and self.featureClassifierVariationCombo.currentText() != "" and self.websiteList.count() != 0:
            self.errorLabel.setText("")
            self.probabilityButton.setDisabled(False)
            self.servicesButton.setDisabled(False)

    def add_website_changed(self):
        """
        Enable the website add button if the input is not empty.

        :return: nothing
        """
        if len(self.addWebsite.text()) > 0:
            self.websiteButton.setDisabled(False)
        else:
            self.websiteButton.setDisabled(True)

    def full_combo_classifier(self):
        """
        Fill up the classifiers combo box.

        :return: nothing
        """
        classifiers = self.app.get_classifiers()
        for cls in classifiers:
            self.classifierCombo.addItem(cls)

    def full_combo_classifier_variation(self):
        """
        Fill up the classifier variations combo box.

        :return: nothing
        """
        # If there are not available classifiers do nothing
        if self.classifierCombo.currentText() != "":
            val = self.classifierCombo.currentText()
            # load variations
            variations = self.app.show_classifier_variations(val)
            for v in variations:
                # if there is the default keyword delete it
                if v.split(" ")[0] == 'default':
                    self.classifierVariationCombo.addItem(" ".join(v.split(" ")[1:]))
                else:
                    self.classifierVariationCombo.addItem(v)

    def full_combo_feature_classifier(self):
        """
        Fill up the features combo box.

        :return: nothing
        """
        features = self.app.get_features()
        for f in features:
            self.featureClassifierCombo.addItem(f)

    def full_combo_feature_classifier_variation(self):
        """
        Fill up the feature variations combo box.

        :return: nothing
        """
        # If there are not available features do nothing
        if self.featureClassifierCombo.currentText() != "":
            val = self.featureClassifierCombo.currentText()
            # Load variations
            variations = self.app.show_feature_variations(val)
            for v in variations:
                # if there is the default keyword delete it
                if v.split(" ")[0] == 'default':
                    self.featureClassifierVariationCombo.addItem(" ".join(v.split(" ")[1:]))
                else:
                    self.featureClassifierVariationCombo.addItem(v)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = Gui()
    sys.exit(app.exec_())
