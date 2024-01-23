import numpy as np
import math


class Dataset:
    """
        Class used for representing annotated datasets consisting of both
        images and labels. Offers methods for fetching certain data, shuffling
        data etc.
    """

    def __init__(self, images, labels):
        """
            :param images: dataset images
            :param labels: dataset labels
        """
        self.images = images
        self.labels = labels

        self.images_mean = np.mean(self.images)
        self.images_std = np.std(self.images)

    def get_item(self, idx):
        """
            Get dataset image/label pair at certain index position.

            :param idx: specific index
            :return: image and label pair
        """
        return self.images[idx], self.labels[idx]

    def get_items(self, start_idx, end_idx):
        """
            Returns dataset images/labels in the defined indices range.

            :param start_idx: start index
            :param end_idx: end index
            :return: images and labels for the defined indices
        """
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(self))

        return self.images[start_idx:end_idx], self.labels[start_idx:end_idx]

    def shuffle(self):
        """
            Shuffles the dataset samples in a random manner.
        """
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        self.images = self.images[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        """
            Return the number of samples in the dataset.

            :return: dataset length
        """
        return len(self.labels)

    def split(self, split_pcts):
        """
            Splits the original dataset into subset datasets which sizes are
            defined by the specified split percentages.

            :param split_pcts: split percentages for new datasets
            :return: dataset splits
        """
        self.shuffle()

        split_datasets = []
        split_datasets_sizes = [round(split_pct * len(self))
                                for split_pct in split_pcts]

        start_idx = 0
        for split_dataset_size in split_datasets_sizes:
            end_idx = start_idx + split_dataset_size

            split_datasets.append(
                Dataset(
                    self.images[start_idx : end_idx],
                    self.labels[start_idx : end_idx]
                )
            )

            start_idx += split_dataset_size

        return split_datasets

    def get_mean_std(self):
        """
            Returns mean and standard deviation of the data samples image pixel
            values.

            :return: mean and standard deviation
        """
        return self.images_mean, self.images_std


class DataLoader():
    """
        Class used for loading data samples from the given dataset. Offers
        some additional methods like data normalization, augmentations etc.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True,
                 normalize=True, augmentations=[]):
        """
            :param dataset: specific dataset
            :param batch_size: batch size
            :param shuffle: whether to shuffle the dataset samples after
                                each epoch
            :param drop_last: whether to drop or include last, possibly not
                                full, batch of data samples
            :param normalize: whether to normalize dataset images
            :param augmentations: list of data augmentations to be used
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.normalize = normalize
        self.augmentations = augmentations

        self.batch_idx = 0
        self.epoch = 1
        self.dataset_size = len(self.dataset)

        if self.drop_last:
            self.num_batches = math.floor(self.dataset_size / self.batch_size)
        else:
            self.num_batches = math.ceil(self.dataset_size / self.batch_size)

        if self.shuffle:
            self.dataset.shuffle()

    def next(self):
        """
            Gets the next batch of data samples.

            :return: batch of images and labels from the dataset
        """
        if self.batch_idx == self.num_batches:
            if self.shuffle:
                self.dataset.shuffle()

            self.batch_idx = 0
            self.epoch += 1

        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx + 1) * self.batch_size

        self.batch_idx += 1

        images, labels = self.dataset.get_items(start_idx, end_idx)

        if self.normalize:
            images_mean, images_std = self.dataset.get_mean_std()
            images = (images - images_mean) / images_std

        for augmentation in self.augmentations:
            images = augmentation.process(images)

        return images, labels

    def current_epoch(self):
        """
            Returns the current epoch number.

            :return: number of the current epoch
        """
        return self.epoch

    def is_last_batch_in_epoch(self):
        """
            Checks if the current batch is the last one in the current epoch.

            :return: whether current batch is the last one in the epoch
        """
        return self.batch_idx == self.num_batches


