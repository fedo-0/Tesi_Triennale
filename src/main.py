import sys
import logging

import pandas as pd

from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser
from trainer.train import run_training_pipeline

setup_logging()
logger = logging.getLogger(__name__)


def prepare_data(input_path: str, output_path: str):
    print(">> prepare_data() chiamata")
    logger.info("Preparing data...")

    df = pd.read_csv(input_path)
    print(df.info())

def run_binclassifier (input_path: str):
    print(">> prepare_data() chiamata")
    logger.info("Running the binary classifier...")
    run_training_pipeline(config_path = "config/dataset.json", csv_path = input_path,
                          outputModel_path="output/models/", outputResults_path="output/results/")


if __name__ == "__main__":
    parser = ArgumentParser("parser")

    parser.register_subcommand(
        subcommand="prepare",
        arguments=["--input", "--output"],
        helps=[
            "The input path for the data.",
            "The output path for the prepared data.",
        ],
        defaults=["resources/datasets/NF-UNSW-NB15-v3.csv", None],
    ).register_subcommand(
        subcommand="binclassifier",
        arguments=["--input", "--output"],
        helps=[
            "The input path for the data.",
            "The output path for the prepared data.",
        ],
        defaults=["resources/datasets/NF-UNSW-NB15-v3.csv", None],
    )

    args = parser.parse_arguments(sys.argv[1:])

    print(f">>> subcommand selezionato: {args}")
    if args.subcommand == "prepare":
        prepare_data(args.input, args.output)
    elif args.subcommand == "binclassifier":
        run_binclassifier(args.input)