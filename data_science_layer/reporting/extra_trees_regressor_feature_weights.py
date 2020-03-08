from data_science_layer.reporting.abstract_report import AbstractReport
from data_science_layer.pipeline.abstract_pipline import AbstractPipeline
from data_science_layer.machine_learning.extra_trees_regressor import ExtraTreesRegressorModel
import pandas as pd
import numpy as np
import pkg_resources


class ExtraTreesRegressorFeatureWeightsReport(AbstractReport):

    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):
        x = pipeline.train
        y = pipeline.train_y

        x['RANDOM_NUMBER'] = np.random.normal(0, 1, x.shape[0])

        et = ExtraTreesRegressorModel()
        et.n_estimators = 100
        et.random_state = pipeline.random_seed
        et.search_models(x, y)

        ft = pd.DataFrame([x.columns.values, et.best_model.feature_importances_],
                          index=['Feature', 'Weight']).transpose().sort_values(by=['Weight'], ascending=False)

        x.drop('RANDOM_NUMBER', axis=1, inplace=True)

        report_df = ft
        folder = ''
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)
        report_df.to_csv(path + pipeline.dataset_tag + '_extra_trees_weights_report.csv')
