from sklearn.dummy import DummyRegressor
from data_science_layer.machine_learning.base_regressor import BaseRegressor


class DummyRegressorModel(BaseRegressor):
    short_name = 'DR'
    sklearn_model = DummyRegressor()

    def __init__(self):
        super().__init__()
        self.set_params()

