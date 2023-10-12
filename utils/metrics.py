import numpy as np
from numpy import ndarray


def MSE(predictions: np.ndarray, targets: np.ndarray) -> ndarray:
    """ Compute the Mean Squared Error (MSE) between predictions and targets.

    The Mean Squared Error is a measure of the average squared difference
    between predicted and actual values. It's a popular metric for regression tasks.

    Formula:
    MSE = (1/n) * Σ (predictions - targets)^2

    where:
    - n is the number of samples
    - Σ denotes the sum
    - predictions are the predicted values by the model
    - targets are the true values
    TODO implement this function. This function is expected to be implemented without the use of loops.

    """
    squared = (predictions - targets) ** 2
    mse = np.mean(squared)

    return mse

