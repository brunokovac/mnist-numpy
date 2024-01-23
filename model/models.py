import os
import numpy as np
from model.layers import FullyConnectedLayer, ReLU


class Model():
    """
        Class representing the model consisting of several layers. Performs
        full forward/backward pass and layer weights manipulations.
    """

    def __init__(self, layers):
        """
            :param layers: model layers
        """
        self.layers = layers

    def forward(self, inputs):
        """
            Performs forward pass through all the layers.

            :param inputs: input data
            :return: output predictions of the model
        """
        out = inputs

        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, learning_rate, loss_grads):
        """
            Performs backward pass through the model and updates the weights
            of all the layers.

            :param learning_rate: learning rate
            :param loss_grads: loss gradients
        """
        grads = loss_grads

        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
            layer.update(learning_rate)

    def save_weights(self, ckpt_dir):
        """
            Saves the weights of the layers as the .npy files.

            :param ckpt_dir: checkpoint directory
        """
        for layer in self.layers:
            layer_weights = layer.get_weights()

            if layer_weights is not None:
                filename = os.path.join(ckpt_dir, f"{layer.name}.npy")
                np.save(filename, layer_weights)

    def load_weights(self, ckpt_dir):
        """
            Loads the weights of the layers from the checkpoint directory.

            :param ckpt_dir: checkpoint directory
        """
        for layer in self.layers:
            ckpt_file = os.path.join(ckpt_dir, f"{layer.name}.npy")

            if os.path.exists(ckpt_file):
                layer.set_weights(np.load(ckpt_file))


class MnistModelSmall(Model):

    def __init__(self):
        super(MnistModelSmall, self).__init__([
            FullyConnectedLayer(channels_in=28 * 28, channels_out=256,
                                name="fc_1"),
            ReLU(name="relu_1"),
            FullyConnectedLayer(channels_in=256, channels_out=10,
                                name="fc_2")
        ])


class MnistModelBig(Model):

    def __init__(self):
        super(MnistModelBig, self).__init__([
            FullyConnectedLayer(channels_in=28 * 28, channels_out=256,
                                name="fc_1"),
            ReLU(name="relu_1"),
            FullyConnectedLayer(channels_in=256, channels_out=128,
                                name="fc_2"),
            ReLU(name="relu_2"),
            FullyConnectedLayer(channels_in=128, channels_out=10,
                                name="fc_3")
        ])
