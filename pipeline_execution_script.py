import pandas as pd
from data_science_pipeline_code.feature_engineering_functions import *
from data_science_layer.pipeline.basic_regressor_pipeline import BasicRegressorPipeline
from data_science_layer.preprocessing.default_scaler import DefaultScaler
import pickle, os
from data_science_layer.scoring.mean_squared_error_scorer import MeanSquaredErrorScorer
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import RandomizedSearchCV, KFold
# from sklearn.metrics import mean_squared_error
# from scipy.stats.distributions import norm
# models_to_try = [GradientBoostingRegressor, RandomForestRegressor, MLPRegressor, LinearRegression]
model_params = [dict(), dict(), dict(), dict()]
import plotly.express as px


def train(dataset):
    script_dir = os.path.dirname(__file__)
    dataset['hr'] = dataset['hr'].astype(int)
    dataset['workingday'] = dataset['workingday'].astype(int)
    dataset['holiday'] = dataset['holiday'].astype(int)
    dataset['dt'] = pd.to_datetime(dataset['dteday'])
    dataset['month'] = dataset['dt'].dt.month
    dataset['year'] = dataset['dt'].dt.year
    dataset['day'] = dataset['dt'].dt.day
    dataset['previous_day_count'] = 0.0
    # Engineer Features (Previous Day's Hourly Rate for daily model, baseline growth models for long term)
    cutoff_date = pd.to_datetime('2012-06-01')
    datasets_per_hour = [get_previous_days_count(dataset, x) for x in range(24)]
    datasets_per_hour = [x.set_index(['dt']) for x in datasets_per_hour]

    # Training and model selection Data Science Pipeline (start of day model)

    train_sets = [x.loc[:cutoff_date] for x in datasets_per_hour]
    test_sets = [x.loc[cutoff_date:] for x in datasets_per_hour]
    pipelines = []
    test_set_results = []
    for idx, x in enumerate(train_sets):
        features = ['atemp', 'workingday', 'holiday', 'windspeed', 'previous_week_count']
        y_val = ['cnt']
        x_train = x[features]
        y_train = x[y_val]
        y_train_scaled = y_train.copy()
        x_train_scaled = x_train.copy()
        current_test = test_sets[idx]
        x_test = current_test[features]
        x_test_scaled = x_test.copy()
        y_test = current_test[y_val]
        y_test_scaled = y_test.copy()
        output_container = y_test.copy()
        output_container.reset_index()
        pipeline = BasicRegressorPipeline()
        pipeline.preprocess_y.append(DefaultScaler())
        pipeline.fit(x_train_scaled, y_train_scaled)
        predictions = pipeline(x_test, y_result=y_test)
        output_container['predicted'] = predictions
        output_container['actual'] = y_test_scaled[y_val].values
        # fig = px.scatter(output_container, x='actual', y='predicted')
        # fig.show()
        with open(script_dir + '/pipeline{}.pkl'.format(idx), "wb") as f:
            pickle.dump(pipeline, f)
        pipelines.append(pipeline)
        test_set_results.append(output_container)
    return pipelines, test_set_results


if __name__ == '__main__':
    # Load in data
    dataset = pd.read_csv('dataset.csv')
    pipelines, test_results = train(dataset)
    results = pd.concat(test_results)
    fig = px.scatter(results, x='actual', y='predicted')
    fig.show()
    print('here')
