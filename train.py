from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(data):
    df = data.to_pandas_dataframe().dropna()
    y_df = df['DEATH_EVENT']
    df = df.drop(['DEATH_EVENT'], axis=1)
    x_df = df
    return x_df, y_df

from azureml.core import Workspace, Dataset

subscription_id = '1b944a9b-fdae-4f97-aeb1-b7eea0beac53'
resource_group = 'aml-quickstarts-139109'
workspace_name = 'quick-starts-ws-139109'

workspace = Workspace(subscription_id, resource_group, workspace_name)

ds = Dataset.get_by_name(workspace, name='dataset')

x, y = clean_data(ds) 

x_train, x_test, y_train, y_test = train_test_split(x, y)


run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("accuracy", np.float(accuracy))

    #To save model
    os.makedirs('outputs', exist_ok = True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
