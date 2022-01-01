#!/usr/bin/env python
"""This script trains a Random Forest
"""
import os
import json
import shutil
import argparse
import logging
import matplotlib.pyplot as plt
import mlflow

import pandas as pd
import numpy as np
from feature_engineering import delta_date_feature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, \
    FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

import wandb


# def delta_date_feature(dates):
#     """
#     Given a 2d array containing dates (in any format recognized by
#     pd.to_datetime), it returns the delta in days
#     between each date and the most recent date in its column
#     """
#     date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
#     return date_sanitized.apply(
#         lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def train(args):
    """Train random forest model
    """
    logger.info("Creating a run of train_random_forest")
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Get the Random Forest configuration and update W&B
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fix the random seed for the Random Forest, so we get reproducible results
    rf_config["random_state"] = args.random_seed

    # Get the artifact path of train and validation artifact from wandb using
    # args.trainval_artifact)
    logger.info("Getting the path of train/valid artifact from wandb ...")
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    logger.info("Generating X (features) and y (target) ...")
    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")  # this removes column "price" from X and puts it into y

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    logger.info("Spliting dataset into train/valid set ...")
    logger.info(f"{100 * args.val_size}% for validation set and "
                f"{100 * (1 - args.val_size)}% for training set")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.val_size,
        stratify=X[args.stratify_by],
        random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline ...")
    sk_pipe, processed_features = get_inference_pipeline(
        rf_config, args.max_tfidf_features)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting training data ...")
    sk_pipe.fit(X_train, y_train)

    # Compute r2 and MAE
    logger.info("Calculating r-square and MAE on validation set ...")
    y_pred = sk_pipe.predict(X_val)
    r_square = sk_pipe.score(X_val, y_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"R-square: {r_square}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # Save model package in the MLFlow sklearn format
    export_path = "random_forest_dir"
    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    ######################################
    # Save the sk_pipe pipeline as a mlflow.sklearn model in the directory
    # "random_forest_dir"
    # HINT: use mlflow.sklearn.save_model
    mlflow.sklearn.save_model(
        sk_pipe,
        export_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=X_val.iloc[:2],
    )
    ######################################

    ######################################
    # Upload the model we just exported to W&B
    # HINT: use wandb.Artifact to create an artifact. Use args.output_artifact
    # as artifact name, "model_export" as type, provide a description and add
    # rf_config as metadata. Then, use the .add_dir method of the artifact
    # instance you just created to add the "random_forest_dir" directory to
    # the artifact, and finally use run.log_artifact to log the artifact
    # to the run.
    logger.info("Uploading RF pipeline artifact to wandb ...")
    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Random Forest pipeline export",
        metadata=rf_config
    )
    artifact.add_dir(export_path)
    run.log_artifact(artifact)
    artifact.wait()
    ######################################

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    ######################################
    logger.info("Saving r-square and MAE score to wandb ...")
    # Here we save r_square under the "r2" key
    run.summary["r2"] = r_square
    # Now log the variable "mae" under the key "mae".
    run.summary["mae"] = mae
    ######################################

    # Upload to W&B the feture importance visualization
    logger.info("Uploading feature importance plot to wandb ...")
    run.log(
        {
          "feature_importance": wandb.Image(fig_feat_imp),
        }
    )


def plot_feature_importance(pipe, feat_names):
    """Plot feature importance

    Parameters
    ----------
        pipe: sklean pipeline
            This is the sklearn pipeline of the Random Forest model.
        feat_names: list
            List of feature names.

    Returns:
        fig_feat_imp: matplotlib.pyplot
            Feature importance plot.
    """
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["classifier_rf"].feature_importances_[:len(feat_names) - 1]

    # NLP importance:
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    nlp_importance = sum(
        pipe["classifier_rf"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)

    # Plot
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    # idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(
        range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()

    return fig_feat_imp


def get_inference_pipeline(rf_config, max_tfidf_features):
    """Mode inference pipeline implementation
    """
    # Let's handle the categorical features first
    # Ordinal categorical are categorical values for which the order is
    # meaningful, for example for room type:
    # 'Entire home/apt' > 'Private room' > 'Shared room'
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    # NOTE: we do not need to impute room_type because the type of the room
    # is mandatory on the websites, so missing values are not possible
    # in production (nor during training). That is not true
    # for neighbourhood_group
    ordinal_categorical_preproc = OrdinalEncoder()

    ######################################
    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
    # 2 - A OneHotEncoder() step to encode the variable
    non_ordinal_categorical_preproc = Pipeline(
        steps=[
            ("simple_imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder())
        ]
    )
    ######################################

    # Let's impute the numerical columns to make sure we handle missing values
    # (note that we do not scale because the RF algorithm does not need that)
    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # A MINIMAL FEATURE ENGINEERING step:
    # we create a feature that represents the number of days passed since the
    # last review.
    # First we impute the missing review date with an old date
    # (because there hasn't been a review for a long time),
    # and then we create a new feature from it,
    date_imputer = Pipeline(
        steps=[
            ("simple_imputer", SimpleImputer(
                strategy='constant', fill_value='2010-01-01')),
            ("function_transformer", FunctionTransformer(
                delta_date_feature, check_inverse=False, validate=False))
        ]
    )

    # Some minimal NLP for the "name" column
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = Pipeline(
        steps=[
            ("simple_imputer", SimpleImputer(
                strategy="constant", fill_value="")),
            ("reshape_to_1d", reshape_to_1d),
            ("tfidf_vecorization", TfidfVectorizer(
                binary=False,
                max_features=max_tfidf_features,
                stop_words='english'))
        ],
    )

    # Let's put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc,
             non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    processed_features = ordinal_categorical + non_ordinal_categorical \
        + zero_imputed + ["last_review", "name"]

    # Create random forest
    random_forest = RandomForestRegressor(**rf_config)

    ######################################
    # Create the inference pipeline. The pipeline must have 2 steps: a step
    # called "preprocessor" applying the ColumnTransformer instance that we
    # saved in the `preprocessor` variable, and a step called "random_forest"
    # with the random forest instance that we just saved in the `random_forest`
    # variable.
    # HINT: Use the explicit Pipeline constructor so you can assign the names
    # to the steps, do not use make_pipeline
    sk_pipe = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier_rf", random_forest)
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic cleaning of dataset")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help=("Artifact containing the training dataset. "
              "It will be split into train and validation")
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help=("Random forest configuration. A JSON dict that will be passed "
              "to the scikit-learn constructor for RandomForestRegressor."),
        default="{}",
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    train(args)
