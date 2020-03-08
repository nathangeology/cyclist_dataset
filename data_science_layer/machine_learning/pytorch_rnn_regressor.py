from data_science_layer.machine_learning.pytorch_nn_regressor import \
    PytorchRegressor
from data_science_layer.machine_learning. \
    not_sk_learn_ml_models.pytorch_rnn import PytorchRnn
from sklearn.metrics.regression import mean_squared_error


class PytorchRnnRegressor(PytorchRegressor):
    sklearn_model = PytorchRnn()
    short_name = 'Pytorch-RNN'

    def __init__(self, model_class=None):
        super().__init__()
        if model_class is None:
            self.sklearn_model = PytorchRnn()
        else:
            self.sklearn_model = model_class()
        self.set_default_model()

    def score_model(self, x, y, kwargs):
        # scorer = self.make_scorer_for_search(kwargs)
        results = self.predict(x)[0]
        running_score = 0
        counter = 0
        for idx in range(x.shape[0]):
            running_score += mean_squared_error(y[idx, :, :],
                                                results[idx, :, :])
            counter += 1
        score = running_score/counter
        return None, score, score
