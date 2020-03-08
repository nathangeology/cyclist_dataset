from imblearn.over_sampling import RandomOverSampler
import pandas as pd


class UpSampler(object):

    sampling_strategy = 'auto'
    return_indices = False
    random_state = 1
    ratio = None

    def fit_sample(self, data, y):
        self._upsampler = RandomOverSampler(sampling_strategy=self.sampling_strategy,
                                            return_indices=self.return_indices,
                                            random_state=self.random_state,
                                            ratio=self.ratio)

        ros_data, ros_y = self._upsampler.fit_sample(data, y)
        data = pd.DataFrame(ros_data, columns=data.columns)
        y = pd.Series(ros_y, name=y.name)
        return data, y
