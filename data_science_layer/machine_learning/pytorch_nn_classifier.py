from data_science_layer.machine_learning.not_sk_learn_ml_models.pytorch_classifier import PytorchCl
from data_science_layer.machine_learning.base_classifier import BaseClassifier


class PytorchClassifier(BaseClassifier):
    sklearn_model = PytorchCl()
    short_name = 'Pytorch-Classifier - funnel NN'

    def __init__(self, model_class=None):
        super().__init__()
        if model_class is None:
            self.sklearn_model = PytorchCl()
        else:
            self.sklearn_model = model_class()
        self.set_default_model()
