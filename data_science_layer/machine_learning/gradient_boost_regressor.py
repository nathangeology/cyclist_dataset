from sklearn.ensemble import GradientBoostingRegressor
from data_science_layer.machine_learning.base_regressor import BaseRegressor


class GradientBoostRegressorModel(BaseRegressor):
    short_name = 'GBR'
    sklearn_model = GradientBoostingRegressor()
    hyper_param_dict = {'n_estimators': [1, 2, 5, 10, 50, 100]}

    def __init__(self):
        super().__init__()
        self.set_params()

