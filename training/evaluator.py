import numpy as np


class ModelEvaluator():
    """
        Class used for evaluating the performance of the model on the
        specific datasets.
    """

    def __init__(self, model):
        """
            :param model: specific model
        """
        self.model = model

        self.confusion_matrix = None #predicted x actual

    def evaluate(self, data_loader):
        """
            Evaluates the specific model on the given dataset.

            :param data_loader: data loader for the dataset
        """
        self.confusion_matrix = None

        for i in range(data_loader.num_batches):
            x, y = data_loader.next()

            x = np.reshape(x, [x.shape[0], -1])

            preds = self.model.forward(x)
            if self.confusion_matrix is None:
                num_classes = preds.shape[-1]
                self.confusion_matrix = np.zeros((num_classes, num_classes))

            preds = np.argmax(preds, axis=1)

            for pred, act in zip(preds, y):
                self.confusion_matrix[pred, act] += 1

    def get_accuracy(self):
        """
            Calculates the accuracy of the model predictions.

            :return: model accuracy
        """
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    def get_precision(self):
        """
            Calculates the precision of the model predictions.

            :return: model precision
        """
        return np.mean(
            np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1))

    def get_recall(self):
        """
            Calculates the recall of the model predictions.

            :return: model recall
        """
        return np.mean(
            np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0))

