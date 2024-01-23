from abc import ABC, abstractmethod
import numpy as np


def one_hot(array, max_value):
    """
        Returns the one-hot encoding matrix based on the given array of labels.

        :param array: array containing the labels
        :param max_value: maximum possible number in the array, also the
                            dimension of the one-hot matrix
        :return: one-hot encoding of the given array
    """
    array_size = array.shape[0]

    one_hot_array = np.zeros((array_size, max_value))
    one_hot_array[np.arange(array_size), array] = 1

    return one_hot_array


class Layer(ABC):
    """
        Class representing a certain layer of the model.
    """

    def __init__(self, name):
        """
            :param name: layer name
        """
        self.name = name

    @abstractmethod
    def forward(self, inputs):
        """
            Performs the forward pass of the layer based on the given inputs.

            :param inputs: inputs data for the given layer
            :return: output values of the layer
        """
        pass

    @abstractmethod
    def backward(self, grads):
        """
            Calculates the gradients for the backward pass through the layer.
            Saves the gradients update for its weights and returns the
            derivative for the previous layer.

            :param grads: gradients of the layer after
            :return: gradients for the previous layer
        """
        pass

    @abstractmethod
    def update(self, learning_rate):
        """
            Updates its weights based on the previously calculated gradients
            from the backward pass.

            :param learning_rate: learning rate
        """
        pass

    @abstractmethod
    def get_weights(self):
        """
            Gets the current weights of the layer.

            :return: current weights
        """
        pass

    @abstractmethod
    def set_weights(self, weights):
        """
            Sets the weights to the given values.

            :param weights: specific weights to set
        """
        pass


class FullyConnectedLayer(Layer):
    """
        Class used for representing fully connected layers. Performs the
        operations on the input data based on Y=XW formula.

    """

    def __init__(self, channels_in, channels_out, name):
        """
            :param channels_in: number of input channels
            :param channels_out: number of output channels
            :param name: layer name
        """
        super(FullyConnectedLayer, self).__init__(name)

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.weights = np.random.normal(
            size=(self.channels_in, self.channels_out),
            loc=0.0, scale=1e-2)

        self.grads_update = 0

    def forward(self, inputs):
        self.x = inputs
        return np.dot(self.x, self.weights)


    def backward(self, grads):
        self.grads_update = np.dot(self.x.T, grads)
        return np.dot(grads, self.weights.T)

    def update(self, learning_rate):
        self.weights -= self.grads_update * learning_rate
        #self.weights = ((1 - (learning_rate * 0.01)/self.x.shape[0]) * self.weights
        #                - self.grads_update * learning_rate)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class ReLU(Layer):
    """
        Layer used for performing rectified linear unit operation.
    """

    def __init__(self, name):
        super(ReLU, self).__init__(name)

    def forward(self, inputs):
        self.x = inputs
        return np.where(self.x > 0, self.x, 0)


    def backward(self, grads):
        return (self.x > 0) * grads

    def update(self, learning_rate):
        pass

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass


class SoftmaxLayer(Layer):
    """
        Class used for calculating softmax operation. Used only for forward
        pass.
    """

    def __init__(self, name):
        super(SoftmaxLayer, self).__init__(name)
        return

    def forward(self, inputs):
        self.x = inputs
        return np.exp(self.x) / np.sum(np.exp(self.x), axis=1)[:, None]

    def backward(self, grads):
        pass

    def update(self, learning_rate):
        pass

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass


class SoftmaxWithCrossEntropyLoss(Layer):
    """
        Softmax layer with additional cross entropy loss implementation. Used
        for training purposes as it also includes the gradient calculations
        for the backward pass.
    """

    def __init__(self, num_classes, name):
        """
            :param num_classes: number of possible dataset classes
            :param name: layer name
        """
        super(SoftmaxWithCrossEntropyLoss, self).__init__(name)

        self.num_classes = num_classes
        self.softmax_fn = SoftmaxLayer(name + "_softmax")

    def forward(self, inputs):
        x, y = inputs

        self.preds = self.softmax_fn.forward(x)
        self.y_one_hot = one_hot(y, self.num_classes)

        correct_class_preds = np.sum(self.preds * self.y_one_hot, axis=1)
        log_values = np.log(correct_class_preds)
        self.loss = -np.mean(log_values)

        return self.loss

    def backward(self, grads=None):
        return self.preds - self.y_one_hot

    def update(self, learning_rate):
        pass

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass

