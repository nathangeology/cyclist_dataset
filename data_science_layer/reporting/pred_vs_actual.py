from data_science_layer.reporting.abstract_report import AbstractReport
from data_science_layer.pipeline.abstract_pipline import AbstractPipeline
import matplotlib.pyplot as plt
import pkg_resources


class ActualVsPredictionPlot(AbstractReport):

    sub_folder = 'reports'

    def report(self, pipeline: AbstractPipeline):
        x = pipeline.train
        y = pipeline.train_y
        pred_y = pipeline(x)
        plt.scatter(y, pred_y)
        plt.suptitle('Predicted vs Actual', fontsize=18, y=1.0)
        plt.xlabel('Actual', fontsize=22)
        plt.ylabel('Predicted', fontsize=22)
        plt.legend( )
        folder = ''
        path = pkg_resources.resource_filename('crcdal', 'cache/' + folder + '/' + self.sub_folder +'/')
        pkg_resources.ensure_directory(path)
        plt.savefig(path + self.dataset_tag + '_Predicted_vs_actual.jpg')
