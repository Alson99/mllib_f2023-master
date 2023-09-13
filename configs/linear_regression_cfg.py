from easydict import EasyDict
from sklearn import datasets

# import numpy as np
cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = datasets.linear_regression_dataset
# cfg.dataframe_path = ''

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1





