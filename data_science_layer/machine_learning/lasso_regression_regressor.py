from sklearn.linear_model import Lasso
from data_science_layer.machine_learning.base_regressor import BaseRegressor


class LassoRegressionRegressorModel(BaseRegressor):
    short_name = 'LaR'
    sklearn_model = Lasso()

    def __init__(self):
        super().__init__()
        self.set_params()







