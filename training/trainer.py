import os.path
import numpy as np
import matplotlib.pyplot as plt


class ModelTrainer():
    """
        Trainer class used for performing the training process of the given
        model on the specific dataset.
    """

    def __init__(self, model, loss_fn, log_every_n_steps=50):
        """
            :param model: specific model to train
            :param loss_fn: loss function
            :param log_every_n_steps: number of steps after which to
                                        log certain training information
        """
        self.model = model
        self.loss_fn = loss_fn
        self.log_every_n_steps = log_every_n_steps

        self.steps = []
        self.loss_values = []
        self.learning_rate_values = []
        self.val_accuracies = []

    def train(self, train_data_loader, val_data_loader, model_evaluator,
              num_steps, learning_rate_scheduler, ckpt_dir):
        """
            Performs the training process on the given datasets.

            :param train_data_loader: data loader for training dataset
            :param val_data_loader: data loader for validation dataset
            :param model_evaluator: model evaluator
            :param num_steps: number of training steps
            :param learning_rate_scheduler: learning rate scheduler
            :param ckpt_dir: directory for saving best checkpoints
        """
        best_val_accuracy = 0.0

        for i in range(num_steps):
            x, y = train_data_loader.next()

            x = np.reshape(x, [x.shape[0], -1])

            preds = self.model.forward(x)
            loss_value = self.loss_fn.forward((preds, y))

            learning_rate = learning_rate_scheduler.get_learning_rate_for_step(i)

            if i % self.log_every_n_steps == 0:
                print(i, learning_rate, loss_value)
                self.steps.append(i)
                self.loss_values.append(loss_value)
                self.learning_rate_values.append(learning_rate)

            loss_grads = self.loss_fn.backward()
            self.model.backward(learning_rate, loss_grads)

            if train_data_loader.is_last_batch_in_epoch():
                model_evaluator.evaluate(val_data_loader)
                val_accuracy = model_evaluator.get_accuracy()

                print()
                print("** validation ** - epoch",
                      train_data_loader.current_epoch(), val_accuracy)
                print()

                self.val_accuracies.append(val_accuracy)

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.model.save_weights(ckpt_dir)

        self.visualize_values(self.steps, self.loss_values,
                              "Training loss",
                              os.path.join(ckpt_dir, "train_loss.png"))
        self.visualize_values(self.steps, self.learning_rate_values,
                              "Learning rate",
                              os.path.join(ckpt_dir, "learning_rate.png"))
        self.visualize_values(range(len(self.val_accuracies)), self.val_accuracies,
                              "Validation accuracy",
                              os.path.join(ckpt_dir, "val_accuracy.png"))

    def visualize_values(self, x, y, title, file_path):
        """
            Visualizes plot graphs of the given values.

            :param x: x-axis values
            :param y: y-axis values
            :param title: plot title
            :param file_path: a file path for saving the plots
        """
        plt.figure()
        plt.title(title)

        plt.plot(x, y)
        plt.savefig(file_path)
