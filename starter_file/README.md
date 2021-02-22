# Predicting Heart Failures

This project aims to predict heart failures based on 12 features of patient information data. Both Automated ML and Hyperparamter tuning are applied and the best performing model of both is deployed.


## Project Set Up and Installation
Download the dataset below and upload it into the Datasets tab in your workspace in Microsoft Azure. 

## Dataset

### Overview
The dataset used is from Kaggle regarding heart failures with 12 features to predict whether an event of death occurs or not. The 12 features are as follows:
- Age
- Anaemia
- Creatinine
- Diabetes
- Injection fraction
- High blood pressure
- Platelets
- Serum Creatinine
- Serum Sodium
- Sex
- Smoking
- Time

### Task
As the prediction task is to predict whether a patient passes away or not, it is a classification task. The above mentioned features will be used to predict the target value (die or not).

### Access
The dataset is uploaded in the AzureML workspace and is accessed with the get_by_name() function of AzureML. 

## Automated ML
In the AutoML configuration the following parameters are applied:

- experiment_timeout_minutes:30, The number of minutes the experiment should stay running.
- primary_metric:'accuracy', The primary metric parameter for optimization.
- n_cross_validations:5, As the dataset is relatively small, a low number of number of cross validations is applied.
- max_concurrent_iterations: 4, To get a dedicated cluster per experiment, the max_concurrent_iterations is set at four, to manage the child runs and when they can be performed. Four max_concurrent_iterations is chosen, as the number of nodes in the cluster is also four.
- training_data=train_data, The train_data is part of the dataset uploaded for this experiment.
- label_column_name='DEATH_EVENT', The target column is death_event, as we want to predict whether a patient passes away.
- featurization='auto', Automatic guardrails and featurization steps part of preprocessing.
- compute_target=cpu_cluster, The ML model is trained on a cluster of Azure virtual machines from Azure Machine Learning Managed Compute. 'STANDARD_D2_V2' with a maximum of 4 nodes in a cpu_cluster is used as compute_target.
- task = 'classification', As the goal is to predict whether someone dies or not, this experiment becomes a classification task type.

### Results
With AutoML the best model has an accuracy of 0.89968 and is an VotingEnsemble model. The parameters can be found in the screenshots folder in this github repositorty in the 'details' screenshot. The model could be improved by changing the autoML parameters (e.g. experiment timeout minutes). 

## Hyperparameter Tuning
The parameters used in the HyperDrive configuration are as follows:

- early_termination_policy: Improving the computational efficiency by terminaly poorly perfoming runs.
- param_sampling: The parameter sampling method indicates the method of searching the hyperparameter space.

### Results
The best model obtained with hyperparameter tuning has an accuracy of 0.86 with regularization strength 4.0 and max iterations 200. To possibly increase the accuracy of the best model, the hyperaparameter search space could be altered. 

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

