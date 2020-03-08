from data_science_layer.reporting.abstract_report import AbstractReport
from data_science_layer.pipeline.abstract_pipline import AbstractPipeline
from data_science_layer.machine_learning.random_forest_classifier import RandomForestClassifierModel
from data_science_layer.machine_learning.gradient_boost_classifier import GradientBoostClassifierModel
from data_science_layer.machine_learning.extra_trees_classifier import ExtraTreesClassifierModel
from data_science_layer.machine_learning.random_forest_regressor import RandomForestRegressorModel
from data_science_layer.machine_learning.gradient_boost_regressor import GradientBoostRegressorModel
from data_science_layer.machine_learning.extra_trees_regressor import ExtraTreesRegressorModel
import pandas as pd
import pkg_resources


class FeatureImportanceReport(AbstractReport):
    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):

        model_list = (RandomForestClassifierModel, GradientBoostClassifierModel, ExtraTreesClassifierModel,
                      RandomForestRegressorModel, GradientBoostRegressorModel, ExtraTreesRegressorModel)

        report_list = []
        for model in pipeline.get_models():
            if isinstance(model, model_list):
                ft = pd.DataFrame([pipeline.train.columns.values, model.best_model.feature_importances_],
                                  index=['Feature', 'Weight']).transpose().sort_values(by=['Weight'], ascending=False)
                ft['Model'] = model.short_name
                report_list.append(ft)

        report_df = pd.concat(report_list)
        folder = ''
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder + '/')
        pkg_resources.ensure_directory(path)
        report_df.to_csv(path + pipeline.dataset_tag + '_model_feature_importance_report.csv')
