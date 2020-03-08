import pandas as pd
from data_science_pipeline_code.feature_engineering_functions import *
from data_science_layer.pipeline.basic_regressor_pipeline import BasicRegressorPipeline
from data_science_layer.preprocessing.default_scaler import DefaultScaler
import pickle, os
model_params = [dict(), dict(), dict(), dict()]
import plotly.express as px


def train(dataset):

    script_dir = os.path.dirname(__file__)

    # Prep Dataset

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
        # Split data into x/y
        x_train = x[features]
        y_train = x[y_val]
        # copy data
        y_train_scaled = y_train.copy()
        x_train_scaled = x_train.copy()

        # prep hold out test data
        current_test = test_sets[idx]
        x_test = current_test[features]
        x_test_scaled = x_test.copy()
        y_test = current_test[y_val]
        y_test_scaled = y_test.copy()

        # get output results DF started
        output_container = y_test.copy()
        output_container.reset_index()

        # create and fit pipeline
        pipeline = BasicRegressorPipeline()
        pipeline.preprocess_y.append(DefaultScaler())
        pipeline.fit(x_train_scaled, y_train_scaled)

        # predict on hold out data
        predictions = pipeline(x_test, y_result=y_test)
        output_container['predicted'] = predictions
        output_container['actual'] = y_test_scaled[y_val].values

        # Write out pipeline to file for later use/deployment
        with open(script_dir + '/pipeline{}.pkl'.format(idx), "wb") as f:
            pickle.dump(pipeline, f)
        pipelines.append(pipeline)
        test_set_results.append(output_container)
    # Return pipelines and results on holdout set
    return pipelines, test_set_results


if __name__ == '__main__':
    # Load in data
    dataset = pd.read_csv('dataset.csv')
    # Train pipelines
    pipelines, test_results = train(dataset)
    # Plot Results
    results = pd.concat(test_results)
    fig = px.scatter(results, x='actual', y='predicted')
    fig.show()

    # You could run a deployment on future data below using pickled pipelines
    # I would refactor the data prep in train into functions, call those then call the pipelines on the new data
