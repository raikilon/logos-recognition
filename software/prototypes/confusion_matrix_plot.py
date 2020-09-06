import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = 25
    for i, j in itertools.product(range(6), range(6)):
        plt.text(j, i, format(cm[i][j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i][j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')


def main():
    class_names = ['DHL', 'FedEx', 'MasterCard', "PayPal", "UPS", "Visa"]
    # Compute confusion matrix
    cnf_matrix = [[28, 0, 11, 0, 1, 0],
                  [0, 25, 5, 0, 0, 0],
                  [0, 0, 50, 0, 0, 0],
                  [0, 0, 7, 32, 0, 1],
                  [0, 0, 3, 2, 25, 0],
                  [0, 0, 6, 0, 0, 44]]

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.show()


if __name__ == "__main__":
    main()
