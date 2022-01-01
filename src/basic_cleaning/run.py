#!/usr/bin/env python
"""Basic data cleaning

Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def clean_data(args):
    """Basic cleaning process
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    logger.info("Created a run for basic cleaning")

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info("Downloaded input artifact from wandb")

    logger.info("Start EDA ...")
    df = pd.read_csv(artifact_local_path, index_col="id")
    min_price, max_price = args.min_price, args.max_price
    idx = df["price"].between(min_price, max_price)
    df = df[idx].copy()
    logger.info("Droped rows with price value outside of range "
                f"[{min_price, max_price}]")

    df["last_review"] = pd.to_datetime(df["last_review"])
    logger.info("Converted last_review datatype to datetime")
    logger.info("Finish EDA ...")

    df.to_csv(args.output_artifact, index=False)
    logger.info(f"Save cleaned dataset to {args.output_artifact}")

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    logger.info("Uploaded output artifact (cleaned dataset) to wandb")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="File name and version of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="File name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price value",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price value",
        required=True
    )

    args = parser.parse_args()

    clean_data(args)
