from abc import ABC, abstractmethod
import numpy as np


class Augmentation(ABC):
    """
        Class representing certain data augmentations. Performs modifications
        on the original data to create new and different data samples each time.
    """

    @abstractmethod
    def process(self, images):
        """
            Processes input images and modifies them.

            :param images: original images
            :return: modified images
        """
        pass


class JitteringAugmentation(Augmentation):
    """
        Augmentation method for adding random jitter noise to the original
        images.
    """

    def __init__(self, max_value):
        """
            :param max_value: maximum added noise value
        """
        self.max_value = max_value

    def process(self, images):
        return images + np.random.normal(
            loc=0.0, scale=self.max_value, size=images.shape)


class ShiftingAugmentation(Augmentation):
    """
        Augmentation method which shifts the image pixels up/down and
        left/right for some randomly generated values each time.
    """

    def __init__(self, max_shift):
        """
            :param max_shift: maximum shifting value for each direction
        """
        self.max_shift = max_shift

    def process(self, images):
        up_down_shift, left_right_shift = np.random.randint(
            low=-self.max_shift, high=self.max_shift, size=2)

        padding_top, padding_bottom = (up_down_shift, 0) \
            if up_down_shift >= 0 else (0, -up_down_shift)
        padding_left, padding_right = (left_right_shift, 0) \
            if left_right_shift >= 0 else (0, -left_right_shift)

        b, h, w = images.shape

        paddings = ((0, 0),
                    (padding_top, padding_bottom),
                    (padding_left, padding_right))
        return np.pad(images, paddings)[:, :h, :w]
