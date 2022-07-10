"""
Creator: Ivanovitch Silva
Date: 30 Jan. 2022
Implement a machine pipeline component that
incorporate preprocessing and train stages.
"""
import argparse
import logging
import os
from functools import partial

import yaml
import tempfile
import mlflow
from mlflow.models import infer_signature

import pandas as pd
import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error, max_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.optimizers as opt

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
LOGGER = logging.getLogger()


class HyperModel(Sequential):
    def __init__(self, config, *args, **kwargs):
        LOGGER.info("Initing HyperModel")
        self.model_config = config
        super().__init__(*args, **kwargs)

    def save(self, export_path):
        self.model.save(export_path)

    def fit(self, x_train, y_train):
        input_shape = (x_train.shape[1], )
        self._build_model(input_shape)
        self._compile_model()

        early_stopping = EarlyStopping(
            patience=20,
            min_delta=5, # 5 dollar variance
            monitor='loss',
            mode='min',
            restore_best_weights=True
        )

        return self.model.fit(
            x_train,
            y_train,
            batch_size=self.model_config.batch_size,
            epochs=self.model_config.epochs,
            callbacks=[WandbCallback(log_weights=True),
                       early_stopping],
            verbose=self.model_config.verbose
        )

    def predict(self, x_val):
        return self.model.predict(x_val)

    @staticmethod
    def build_optimizer(optimizer, learning_rate):
        optimizer_map = {
            'adadelta': opt.Adadelta,
            'adagrad': opt.Adagrad,
            'adam': opt.Adam,
            'adamax': opt.Adamax,
            'ftrl': opt.Ftrl,
            'nadam': opt.Nadam,
            'rmsprop': opt.RMSprop,
            'sgd': opt.SGD
        }
        return optimizer_map[optimizer](learning_rate=learning_rate)

    def _build_model(self, input_shape):
        model = Sequential()

        for i in range(self.model_config.num_layers):
            if i == 0:
                model.add(
                    Dense(
                        units=self.model_config.units,
                        input_shape=input_shape,
                        kernel_initializer=self.model_config.kernel_initializer
                    )
                )
            else:
                model.add(
                    Dense(
                        units=self.model_config.units,
                        kernel_initializer=self.model_config.kernel_initializer
                    )
                )

            if self.model_config.batch_normalization:
                model.add(BatchNormalization())

            model.add(Activation(self.model_config.hidden_activation))
            model.add(Dropout(self.model_config.dropout))

        model.add(
            Dense(
                1,
                activation=self.model_config.output_activation
            )
        )
        self.model = model

    def _compile_model(self):
        optimizer = HyperModel.build_optimizer(self.model_config.optimizer,
                                               self.model_config.learning_rate)
        loss = self.model_config.loss
        metrics = list(self.model_config.metrics)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Custom Transformer that extracts columns passed as argument to its constructor


class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self.feature_names = feature_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        return X[self.feature_names]

# transform numerical features


class NumericalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes a model parameter as its argument
    # model 0: minmax
    # model 1: standard
    # model 2: without scaler
    def __init__(self, model=0, colnames=None):
        self.model = model
        self.colnames = colnames

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # return columns names after transformation
    def get_feature_names(self):
        return self.colnames

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # update columns name
        self.colnames = df.columns.tolist()

        # minmax
        if self.model == 0:
            scaler = MinMaxScaler()
            # transform data
            df = scaler.fit_transform(df)
        elif self.model == 1:
            scaler = StandardScaler()
            # transform data
            df = scaler.fit_transform(df)
        else:
            df = df.values

        return df


def process_args(args):

    run = wandb.init(job_type="train")

    LOGGER.info("Downloading and reading train artifact")
    local_path = run.use_artifact(args.train_data).file()
    df_train = pd.read_csv(local_path)

    # Spliting train.csv into train and validation dataset
    LOGGER.info("Spliting data into train/val")
    # split-out train/validation and test dataset
    x_train, x_val, y_train, y_val = train_test_split(df_train.drop(labels=args.target, axis=1),
                                                      df_train[args.target],
                                                      test_size=args.val_size,
                                                      random_state=args.random_seed,
                                                      shuffle=True,
                                                      stratify=df_train[args.stratify])

    LOGGER.info("x train: {}".format(x_train.shape))
    LOGGER.info("y train: {}".format(y_train.shape))
    LOGGER.info("x val: {}".format(x_val.shape))
    LOGGER.info("y val: {}".format(y_val.shape))

    LOGGER.info("Removal Outliers")
    # temporary variable
    x = x_train.select_dtypes(["int64", "float"]).copy()

    # identify outlier in the dataset
    lof = LocalOutlierFactor()
    outlier = lof.fit_predict(x)
    mask = outlier != -1

    LOGGER.info("x_train shape [original]: {}".format(x_train.shape))
    LOGGER.info("x_train shape [outlier removal]: {}".format(
        x_train.loc[mask, :].shape))

    # dataset without outlier, note this step could be done during the preprocesing stage
    x_train = x_train.loc[mask, :].copy()
    y_train = y_train[mask].copy()

    # Pipeline generation
    LOGGER.info("Pipeline generation")

    # Get the configuration for the pipeline
    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)

    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config["neural_network"])

    # Categrical features to pass down the categorical pipeline
    categorical_features = x_train.select_dtypes("object").columns.to_list()

    # Numerical features to pass down the numerical pipeline
    numerical_features = x_train.select_dtypes(
        ["int64", "float"]).columns.to_list()

    # Defining the steps in the categorical pipeline
    categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(categorical_features)),
                                           # ('cat_encoder','passthrough'
                                           ('cat_encoder', OneHotEncoder(
                                               sparse=False, drop="first"))
                                           ]
                                    )
    # Defining the steps in the numerical pipeline
    numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(numerical_features)),
                                         ('num_transformer', NumericalTransformer(model_config["numerical_pipe"]["model"],
                                                                                  colnames=numerical_features))
                                         ]
                                  )

    # Combining numerical and categorical piepline into one full big pipeline horizontally
    # using FeatureUnion
    full_pipeline_preprocessing = FeatureUnion(transformer_list=[('cat_pipeline', categorical_pipeline),
                                                                 ('num_pipeline',
                                                                  numerical_pipeline)
                                                                 ]
                                               )

    # The full pipeline
    pipe = Pipeline(steps=[('full_pipeline', full_pipeline_preprocessing),
                           ('regressor', HyperModel(wandb.config))
                           ]
                    )

    # training
    LOGGER.info("Training")
    pipe.fit(x_train, y_train)

    # predict
    LOGGER.info("Infering")
    predict = pipe.predict(x_val)

    # Evaluation Metrics
    LOGGER.info("Evaluation metrics")
    # Metric: Mean Absolute Error
    mae = mean_absolute_error(y_val, predict)
    run.summary["MAE"] = mae

    # Metric: Max error
    max_err = max_error(y_val, predict)
    run.summary["Max Error"] = max_err

    # Metric: Explained Variance Score
    evs = explained_variance_score(y_val, predict)
    run.summary["Explained Variance Score"] = evs

    # Export if required
    if args.export_artifact != "null":
        export_model(run, pipe, x_val, predict, args.export_artifact)


def export_model(run, pipe, x_val, val_pred, export_artifact):

    # Infer the signature of the model
    signature = infer_signature(x_val, val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        mlflow.keras.save_model(
            pipe['regressor'],  # our pipeline
            os.path.join(export_path, 'regression'),  # Path to a directory for the produced package
            keras_module=tf.keras,
            signature=signature,  # input and output schema
            input_example=x_val.iloc[:2],  # the first few examples
        )

        mlflow.sklearn.save_model(
            pipe['full_pipeline'],  # our pipeline
            os.path.join(export_path, 'full_pipeline'),  # Path to a directory for the produced package
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,  # input and output schema
            input_example=x_val.iloc[:2],  # the first few examples
        )

        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="Neural Network pipeline export",
        )

        # NOTE that we use .add_dir and not .add_file
        # because the export directory contains several
        # files
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a YML file containing the configuration for the random forest",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for the random number generator.",
        required=False,
        default=42
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size for the validation set as a fraction of the training set",
        required=False,
        default=0.3
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="Name of a column to be used for stratified sampling. Default: 'null', i.e., no stratification",
        required=False,
        default="null",
    )

    parser.add_argument(
        "--target",
        type=str,
        help="Name of the target column with the output values or classes to be predicted",
        required=True
    )

    ARGS = parser.parse_args()

    process_args(ARGS)
