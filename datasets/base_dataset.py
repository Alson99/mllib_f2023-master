import numpy as np
from abc import ABC, abstractmethod


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    @property
    @abstractmethod
    def test(self):
        pass

    def _divide_into_sets(self):
        # TODO define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test

        # Calculate the number of examples for each set based on percentages
        num_try = len(self.targets)
        num_train_try = int(num_try * self.train_set_percent)
        num_valid_try = int(num_try * self.valid_set_percent)
        # num_test_try = num_try - num_train_try - num_valid_try

        # Shuffle the data randomly before dividing
        indices = np.arange(num_try)
        np.random.shuffle(indices)

        # Divide the data into training, validation and test sets
        self.inputs_train = self.inputs[indices[:num_train_try]]
        self.targets_train = self.targets[indices[:num_train_try]]

        self.inputs_valid = self.inputs[indices[num_train_try:num_train_try + num_valid_try]]
        self.targets_valid = self.targets[indices[num_train_try:num_train_try + num_valid_try]]

        # self.inputs_test = self.inputs[indices[num_try + num_valid_try:]]
        # self.targets_test = self.targets[indices[num_train_try + num_valid_try:]]
