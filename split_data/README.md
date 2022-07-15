# Instructions

This module is responsible to split the clean data saved in the last step (preprocessing) as two new artifact in Weights and Biases projects (`train_data.csv` and `test_data.csv`).

To split the data, we are using the `train_test_split` function from `sklearn.model_selection` package. That functions allow us to split our dataset into two new datasets, that will be used to train and test our model, passing specific parameteres like `random_state`, where you can control how the dataset will be splitted and shuffled and make the project more reproducible. Theres are also a `test_size` parameter, where you can define the size of the test dataset (0 to 1, where 0.3 means 30% of the dataset will be used to test our model) and `stratify` parameter, used when you want to keep the target variable distribution the same in both the training and test datasets.

## Run Steps

Here is an example to run this step. Inside the project root directory, run:

```bash
mlflow run . -P hydra_options="main.execute_steps='segregate'"
```

Note: Only use `stratify` parameter if you have a categorical column with 2 or more classes. Remember that in binary categorical data you have two classes: Positive and Negative.
