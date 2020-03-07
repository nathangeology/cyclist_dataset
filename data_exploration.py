import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


if __name__ == '__main__':
    dataset = pd.read_csv('dataset.csv')
    # fig = px.box(dataset, x='hr', y='cnt')
    # fig.show()
    dataset['hr'] = dataset['hr'].astype(int)
    peak_hr = dataset[dataset['hr'] == 17]
    fig = px.box(peak_hr, x='season', y='cnt')
    fig.show()
    print('here')