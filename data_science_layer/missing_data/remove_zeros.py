import pandas as pd
import numpy as np


class RemoveZeros(object):
    @classmethod
    def fit_transform(cls, data):
        obj = cls( )
        return obj._process_data(data)

    def transform(self, data):
        return self._process_data(data)

    def fit(self, data, y=None):
        return self._process_data(data)

    def _process_data(self, data, y=None):
        df = self._check_input(data)
        df = df.replace(0, np.nan)
        a_filter = df.T.isna( ).any( )
        df = df.dropna( )
        df = self._check_output(data, df)
        return df, a_filter

    def _check_input(self, data):
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
        return df

    def _check_output(self, input_data, output):
        if not isinstance(input_data, pd.DataFrame):
            output = output.values
        return output
