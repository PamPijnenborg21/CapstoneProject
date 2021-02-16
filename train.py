from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset, Datastore
from azureml.data.datapath import DataPath



run = Run.get_context()


from azureml.core import Workspace, Dataset

subscription_id = '5a4ab2ba-6c51-4805-8155-58759ad589d8'
resource_group = 'aml-quickstarts-139017'
workspace_name = 'quick-starts-ws-139017'

workspace = Workspace(subscription_id, resource_group, workspace_name)


ds = Dataset.get_by_name(workspace, name='dataset')
print(' this is ds')
print(ds)


def clean_data(data):
    df = data.to_pandas_dataframe().dropna()
    y_df = df['DEATH_EVENT']
    df.drop(['DEATH_EVENT'], inplace=True, axis=1)
    x_df = df
    return x_df, y_df
    
x, y = clean_data(ds)
print(' x : ' +str(x))
print(' y : ' +str(y))

# Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x, y)
print(' x train: ' +str(x_train))
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
    run.log("Accuracy", np.float(accuracy))

    #to save model
    os.makedirs('outputs', exist_ok = True)
    joblib.dump(model,'outputs/model.joblib')


if __name__ == '__main__':
    main()