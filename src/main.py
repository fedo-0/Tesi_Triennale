import sys
import logging

import pandas as pd

from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser
from trainer.train import run_training_pipeline

setup_logging()
logger = logging.getLogger(__name__)


def prepare_data(input_path: str, output_path: str):
    logger.info("Preparing data...")

    df = pd.read_csv(input_path)

    numeric_cols = [
        "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
        "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
        "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES",
        "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES",
        "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
        "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT",
        "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
        "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
        "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_TTL_ANSWER",
        "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG",
        "SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
        "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV"
    ]

    # 1) calcolo percentili
    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    percentili = df[numeric_cols].quantile(q=qs).T
    percentili.columns = [f"{int(q*100)}%" for q in qs]

    # 2) min e max
    descr = df[numeric_cols].agg(['min', 'max']).T
    descr = descr.rename(columns={'min': 'Min', 'max': 'Max'})

    # 3) unisco in stats
    stats = pd.concat([descr, percentili], axis=1)
    print("=== Statistiche descrittive e percentili ===")
    print(stats)

    # 4) calcolo outlier secondo IQR
    outlier_info = []
    for col in numeric_cols:
        Q1 = stats.loc[col, '25%']
        Q3 = stats.loc[col, '75%']
        IQR = Q3 - Q1
        low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        total = len(df)
        n_outliers = ((df[col] < low) | (df[col] > high)).sum()
        pct_outliers = n_outliers / total * 100

        outlier_info.append({
            'feature': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'low_thr': low,
            'high_thr': high,
            'n_outliers': n_outliers,
            'pct_outliers': pct_outliers
        })

    outliers_df = pd.DataFrame(outlier_info).set_index('feature')
    print("\n=== Outlier (IQR rule) per feature ===")
    print(outliers_df[['n_outliers', 'pct_outliers', 'low_thr', 'high_thr']])

    # print(df.info())

def run_binclassifier (input_path: str, model_size: str):
    print(">> prepare_data() chiamata")
    logger.info("Running the binary classifier...")
    run_training_pipeline(
        config_path = "config/dataset.json", 
        csv_path = input_path,
        outputModel_path="models/", 
        outputResults_path="results/",
        model_size=model_size
    )


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
        arguments=["--input", "--model-size"],
        helps=[
            "The input path for the data.",
            "The model size: small, medium, or large.",
        ],
        defaults=["resources/datasets/NF-UNSW-NB15-v3.csv", "small"],
    )

    args = parser.parse_arguments(sys.argv[1:])

    print(f">>> subcommand selezionato: {args}")
    if args.subcommand == "prepare":
        prepare_data(args.input, args.output)
    elif args.subcommand == "binclassifier":
        run_binclassifier(args.input, args.model_size)