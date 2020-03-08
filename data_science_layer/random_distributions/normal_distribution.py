from data_science_layer.random_distributions.abstractdistribution import AbstractDistribution
from scipy.stats import norm


class NormalDistribution(AbstractDistribution):
    """Class to help with fitting and creating normally distributed random feature examples"""

    @classmethod
    def generate_random_examples(cls, *, data, num_of_examples):
        """Loops through each feature column, fits a normal distribution and then randomly generates new examples"""

        # TODO: Handle non-numeric input cols
        obj = cls(data=data, num_of_examples=num_of_examples)
        obj._check_type(data)
        obj._create_output( )

        for column in range(data.shape[1]):
            """ for each feature column, generate a normal distribution
            and then generate additional random examples within the distribution"""
            col_data = obj._get_column(column)
            current_col_data = obj._generate_new_examples(col_data)
            obj._set_column(column=column, data=current_col_data)
        return obj._output

    def _generate_new_examples(self, column):
        distribution_vals = norm.fit(column)
        new_examples = norm.rvs(loc=distribution_vals[0],
                                scale=distribution_vals[1],
                                size=self._num_examples)
        return new_examples
