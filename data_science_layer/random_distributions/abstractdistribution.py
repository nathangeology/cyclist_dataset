from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class AbstractDistribution(ABC):
    _is_pandas = False
    _num_examples = 0
    _input_data = None
    _output = None

    def __init__(self, *, data, num_of_examples):
        self._num_examples = num_of_examples
        self._input_data = data

    @classmethod
    @abstractmethod
    def generate_random_examples(cls, *, data, num_of_examples):
        raise (NotImplementedError('Do not use abstract method directly'))

    def _check_type(self, input):
        if isinstance(input, pd.DataFrame):
            self._is_pandas = True
            return
        if not isinstance(input, np.ndarray):
            raise (TypeError('Unknown data type for generating new examples'))

    def _create_output(self):
        output_vals = np.zeros([self._num_examples, self._input_data.shape[1]])
        if self._is_pandas:
            output_vals = pd.DataFrame(data=output_vals,
                                       index=range(self._num_examples),
                                       columns=self._input_data.columns)
        self._output = output_vals

    def _get_column(self, column):
        if self._is_pandas:
            return self._input_data.iloc[:, column]
        else:
            return self._input_data[:, column]

    def _set_column(self, *, column, data):
        if self._is_pandas:
            self._output.iloc[:, column] = data
        else:
            self._output[:, column] = data
