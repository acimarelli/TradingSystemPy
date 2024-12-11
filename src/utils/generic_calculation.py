import numpy as np
import pandas as pd


class GenericCalculation:

    @staticmethod
    def portfolio_annualised_performance(weights: np.array, mean_returns: np.array, cov_matrix: np.matrix):
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return std, returns

