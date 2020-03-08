from sklearn.ensemble import ExtraTreesClassifier
from data_science_layer.machine_learning.base_classifier import BaseClassifier


class ExtraTreesClassifierModel(BaseClassifier):
    short_name = 'ETC'
    sklearn_model = ExtraTreesClassifier()
    hyper_param_dict = {'n_estimators': [1, 2, 5, 10, 50, 100]}

    def __init__(self):
        super().__init__()
        self.set_params()

