from numpy import arange, argmax
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

class GraphPlot:
    def __init__(self):
        pass

    def showConfusionMatrix(self, batches_classes, predictions, class_names, title):
        cm = confusion_matrix(y_true=batches_classes, y_pred=argmax(predictions, axis=-1))
        self.plotConfusionMatrix(cm=cm, classes=class_names, title=title)

    def plotConfusionMatrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def showAccuracy(self, history, epochs=None):
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        if epochs is None:
            epochs = range(1, len(accuracy) + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        ax1.plot(loss, color='b', label="Training loss")
        ax1.plot(val_loss, color='r', label="validation loss")
        ax1.set_xticks(arange(1, epochs, 1))
        ax1.set_yticks(arange(0, 1, 0.1))

        ax2.plot(accuracy, color='b', label="Training accuracy")
        ax2.plot(val_accuracy, color='r', label="Validation accuracy")
        ax2.set_xticks(arange(1, epochs, 1))

        legend = plt.legend(loc='best', shadow=True)
        plt.tight_layout()
        plt.show(block=False)

    def plotImages(self, images_arr):
        fig, axes = plt.subplots(1, 10, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)