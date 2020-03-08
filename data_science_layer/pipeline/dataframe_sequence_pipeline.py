from data_science_layer.pipeline.basic_regressor_pipeline import \
    BasicRegressorPipeline
from data_science_layer.machine_learning.pytorch_rnn_regressor import \
    PytorchRnnRegressor
from data_science_layer.data_split.dataframe_seq_split import\
    DataFrameSequenceSplitter
import numpy as np


class DataFrameSequencePipeline(BasicRegressorPipeline):
    _use_2d_preprocessing = False

    def __init__(self):
        super().__init__()
        self.splitter = DataFrameSequenceSplitter()
        self._feature_selection = None
        self._xy_generator_class = None

    def _fit_preprocessing(self, processor):
        pass
        # self._fit_preprocessor_to_table(processor)
        # self.test = self._apply_preprocessor_to_table(processor, self.test, y=None)

    def _transform_preprocessing(self, processor, data, y_data=None):
        temp_table, tables, rows = self._transform_table(data)
        output = self._apply_preprocessor_to_table(
            processor, temp_table, y_data)
        return self._untransform_table(output, tables, rows)

    def _apply_preprocessor_to_table(self, processor, table, y):
        temp_table, tables, rows = self._transform_table(table)
        table = processor.transform(temp_table, y=y)
        table = self._untransform_table(table, tables, rows)
        return table

    def _transform_table(self, table):
        table_count = table.shape[0]
        rows_count = table.shape[1]
        output = table.reshape((
            table.shape[0] * table.shape[1], table.shape[2]))
        return output, table_count, rows_count

    def _fit_preprocessor_to_table(self, processor):
        temp_train, table_count, rows_count = self._transform_table(self.train)
        table = processor.fit_transform(temp_train, y=self.train_y)
        self.train = self._untransform_table(table, table_count, rows_count)

    def _untransform_table(self, table, table_count, rows_count):
        return table.values.reshape(
            (table_count, rows_count, table.shape[1]))

    def _find_test_score(self, ml_model, **kwargs):
        score = ml_model.score_model(self.test, self.test_y, kwargs)[1]
        self.models_results.append(score)
        print('test_set_' + str('RMSE') + '_score: ' + str(score))

    def define_default_models(self):
        self._ml_models = []
        self.add_ml_model(ml=PytorchRnnRegressor())

    def format_inputs(self, data, training_y):
        out_x = np.stack(data)
        out_y = np.stack(training_y)
        return out_x, out_y

