#!/usr/bin/env python
"""This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def split_train_val_test(args):
    """Train test split
    """
    logger.info("Creating a run for splitting train/test/valid set")
    run = wandb.init(job_type="data_split")
    run.config.update(args)

    # Download input artifact. This will also note that this script is using
    # this particular version of the artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files
    for df, k in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {k}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w") as fp:

            df.to_csv(fp.name, index=False)

            log_artifact(
                f"{k}_data.csv",
                f"{k}_data",
                f"{k} split of dataset",
                fp.name,
                run,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument(
        "input",
        type=str,
        help="Input artifact to split"
    )

    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split"
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default='none',
        required=False
    )

    args = parser.parse_args()

    split_train_val_test(args)
