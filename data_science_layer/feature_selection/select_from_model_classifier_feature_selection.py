from data_science_layer.feature_selection.abstract_feature_selector import AbstractFeatureSelector
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from crcdal.data_science_layer.machine_learning.logistic_regression_classifier import LogisticRegressionClassifierModel


class SelectFromModelClassifierSelector(AbstractFeatureSelector):

    @classmethod
    def select_features(cls, x, y):
        obj = cls()
        return obj.select_features_from_model(x, y)

    def select_features_from_model(self, x, y):

        selector = SelectFromModel(estimator=LogisticRegression().fit(x, y), threshold=self.threshold,
                                   prefit=self.prefit, norm_order=self.norm_order)
        selector.fit_transform(x, y)
        features = selector.get_support(indices=True)
        self.best_features = [column for column in x.columns[features]]
        x_select = self.select_features_in_test_set(x)

        return x_select
