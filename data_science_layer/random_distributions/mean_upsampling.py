from data_science_layer.random_distributions.abstractdistribution import AbstractDistribution
import pandas as pd


class MeanUpsampling(AbstractDistribution):
    @classmethod
    def generate_random_examples(cls, *, data, num_of_examples):
        means = data.mean(axis=1)
        output = pd.concat([means] * num_of_examples, axis=1)
        return output.T
