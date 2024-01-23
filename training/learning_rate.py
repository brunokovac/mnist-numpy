from abc import ABC, abstractmethod


class LearningRateSchedule(ABC):
    """
        Class representing certain learning rate schedules.
    """

    @abstractmethod
    def get_learning_rate_for_step(self, step):
        """
            Returns the current learning rate value for the specific training
            step.

            :param step: current training iterations step
            :return: current learning rate value
        """
        pass


class ConstantLearningRate(LearningRateSchedule):
    """
        Constant learning rate scheduler which maintains the same learning
        rate value throughout the whole training process.
    """

    def __init__(self, learning_rate):
        """
            :param learning_rate: constant learning rate value
        """
        self.learning_rate = learning_rate

    def get_learning_rate_for_step(self, step):
        return self.learning_rate


class PolynomialLearningRate(LearningRateSchedule):
    """
        Class used for learning rate schedule that uses polynomial decay of
        the values between starting and end learning rate.
    """

    def __init__(self, start_learning_rate, end_learning_rate, power,
                 max_steps):
        """
            :param start_learning_rate: starting learning rate
            :param end_learning_rate: ending learning rate
            :param power: exponent power for the speed of learning rate decay
            :param max_steps: maximum number of training steps
        """
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.max_steps = max_steps

    def get_learning_rate_for_step(self, step):
        return ((self.start_learning_rate - self.end_learning_rate) \
                    * (1 - step / self.max_steps) ** self.power) \
                + self.end_learning_rate