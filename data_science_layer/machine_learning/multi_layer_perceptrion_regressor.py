from sklearn.neural_network import MLPRegressor
from data_science_layer.machine_learning.base_regressor import BaseRegressor


class MultiLayerPerceptronRegressorModel(BaseRegressor):
    short_name = 'MLPR'
    sklearn_model = MLPRegressor()
    hyper_param_dict = {'hidden_layer_sizes': [[100,100], [300,300], [500,500]]}

    def __init__(self):
        super().__init__()
        self.set_params()




