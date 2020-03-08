from sklearn.dummy import DummyClassifier
from data_science_layer.machine_learning.base_classifier import BaseClassifier


class DummyClassifierModel(BaseClassifier):
    short_name = 'DC'
    sklearn_model = DummyClassifier()

    def __init__(self):
        super().__init__()
        self.set_params()
