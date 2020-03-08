from data_science_layer.random_distributions.abstractdistribution import AbstractDistribution
import pandas as pd


class ReplicateExamples(AbstractDistribution):
    @classmethod
    def generate_random_examples(cls, *, data, num_of_examples):
        number_of_replications = num_of_examples / len(data)
        if not isinstance(number_of_replications, int):
            number_of_replications = int(number_of_replications) + 1
        output = pd.concat([data] * number_of_replications, ignore_index=True)
        return output.loc[:num_of_examples - 1]
