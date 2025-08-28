import sys
import logging

import pandas as pd

from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser

from data.bin_preprocessing import preprocess_dataset_binary
from data.multi_preprocessing import preprocess_dataset_multiclass
from data.preprocessing_transformer import preprocess_dataset_transformer
from trainer.trainer_bin import main_pipeline_bin
from trainer.trainer_multiclass import main_pipeline_multiclass
from trainer.trainer_transformer import main_pipeline_transformer
from data.split import clean_and_split_dataset

setup_logging()
logger = logging.getLogger(__name__)

def split (input_path: str, output_dir: str):
    logger.info("Pulizia e Divisione del dataset in corso...")
    
    clean_and_split_dataset(
        dataset_path=input_path,
        config_path="config/dataset.json",
        output_dir=output_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        window_size=10,
        label_col='Label',
        attack_col='Attack',
        min_samples_per_class=10000
    )

    logger.info("✅ Clean e Split del dataset completato con successo!")

def prepare_data(input_path: str, output_dir: str):
    logger.info("Preparing data...")
    
    """
    logger.info(f"Numero totale di righe: {len(df)}")
    all_cols = [
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
    """
    try:
        # esecuzione del preprocessing multiclasse
        df_train, df_val, df_test, scaler, freq_mappings, label_encoder, class_mapping = preprocess_dataset_multiclass(
            clean_split_dir=input_path,
            config_path="config/dataset.json",
            output_dir=output_dir
        )
        """
        train_data, val_data, test_data, scaler, label_encoder, metadata = preprocess_dataset_transformer(
            dataset_path=input_path,
            config_path="config/hyperparameters_transformer.json",
            output_dir=output_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            min_window_size=10,
            max_window_size=30,
            min_samples_per_class=10000,
            benign_undersample_ratio=0.5
        )"""
        
        logger.info("✅ Preprocessing completato con successo!")
        
    except FileNotFoundError as e:
        logger.info(f"❌ Errore: File non trovato - {e}")
    except ValueError as e:
        logger.info(f"❌ Errore nei dati: {e}")
    except Exception as e:
        logger.info(f"❌ Errore imprevisto: {e}")

def run_binclassifier (model_size: str):
    logger.info("Preparando il classificatore binario...")
    main_pipeline_bin()
    logger.info("✅ Training completato con successo!")

def run_multiclassifier (model_size:str):
    logger.info("Preparando il classificatore multiclasse...")
    main_pipeline_multiclass(model_size="small")
    logger.info("✅ Training completato con successo!")

def run_transformer (model_size:str):
    logger.info("Preparando il trasformer per la classificazione multiclasse...")
    main_pipeline_transformer(model_size="small")
    logger.info("✅ Training completato con successo!")

if __name__ == "__main__":
    parser = ArgumentParser("parser")

    parser.register_subcommand(
        subcommand="prepare",
        arguments=["--input", "--output"],
        helps=[
            "The clean split directory.",
            "The output directory for the prepared data.",
        ],
        defaults=["resources/datasets", "resources/datasets"],
    ).register_subcommand(
        subcommand="split",
        arguments=["--input", "--output"],
        helps=[
            "The input path for the data.",
            "The output directory for the prepared data.",
        ],
        defaults=["resources/datasets/dataset_ton_v3.csv", "resources/datasets"],
    ).register_subcommand(
        subcommand="runbinary",
        arguments=["--model-size"],
        helps=[
            "The model size: small, medium, or large.",
        ],
        defaults=["small"],
    ).register_subcommand(
        subcommand="runmulti",
        arguments=["--model-size"],
        helps=[
            "The model size: small, medium, or large.",
        ],
        defaults=["small"],
    ).register_subcommand(
        subcommand="runtrans",
        arguments=["--model-size"],
        helps=[
            "The model size: small, medium, or large.",
        ],
        defaults=["small"]
    )

    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
        prepare_data(args.input, args.output)
    elif args.subcommand == "split":
        split(args.input, args.output)
    elif args.subcommand == "runbinary":
        run_binclassifier(args.model_size)
    elif args.subcommand == "runmulti":
        run_multiclassifier(args.model_size)
    elif args.subcommand == "runtrans":
        run_transformer(args.model_size)