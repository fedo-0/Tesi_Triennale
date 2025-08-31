import sys
import logging

from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser

from data.bin_preprocessing import preprocess_dataset_binary
from data.multi_preprocessing import preprocess_dataset_multiclass
from data.transformer_preprocessing import preprocess_dataset_transformer
from trainer.trainer_bin import main_pipeline_bin
from trainer.trainer_multiclass import main_pipeline_multiclass
from trainer.trainer_transformer import main_pipeline_transformer
from data.split import clean_and_split_dataset

setup_logging()
logger = logging.getLogger(__name__)

def split (input_path: str, output_dir: str):
    logger.info("Pulizia e Divisione del dataset in corso...")
    min_samples=10000
    if (input_path=="resources/datasets/NF-UNSW-NB15-v3.csv"):
        min_samples=2000
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
        min_samples_per_class=min_samples
    )

    logger.info("✅ Clean e Split del dataset completato con successo!")

def prepare_mlp(input_path: str, output_dir: str):
    logger.info("Preparing data for MLP ARCHITECTURE...")
    try:
        # esecuzione del preprocessing multiclasse MLP
        df_train, df_val, df_test, scaler, freq_mappings, label_encoder, class_mapping = preprocess_dataset_multiclass(
            clean_split_dir=input_path,
            config_path="config/dataset.json",
            output_dir=output_dir
        )
        logger.info("✅ Preprocessing completato con successo!")
    except FileNotFoundError as e:
        logger.info(f"❌ Errore: File non trovato - {e}")
    except ValueError as e:
        logger.info(f"❌ Errore nei dati: {e}")
    except Exception as e:
        logger.info(f"❌ Errore imprevisto: {e}")

def prepare_transformer(input_path: str, output_dir: str):
    logger.info("Preparing data for TRANSFORMER ARCHITECTURE...")
    try:
        # esecuzione del preprocessing multiclasse TRANSFORMER
        preprocess_dataset_transformer(
            clean_split_dir=input_path,
            config_path="config/dataset.json",
            output_dir=output_dir,
            label_col='Label',
            attack_col='Attack',
            sequence_length=64,
            sequence_stride=32,
            min_freq_categorical=10,
            max_vocab_size=10000
        )
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
        subcommand="prepMLP",
        arguments=["--input", "--output"],
        helps=[
            "The clean split directory.",
            "The output directory for the prepared data.",
        ],
        defaults=["resources/datasets", "resources/datasets"],
    ).register_subcommand(
        subcommand="prepTRANS",
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

    if args.subcommand == "prepMLP":
        prepare_mlp(args.input, args.output)
    elif args.subcommand == "prepTRANS":
        prepare_transformer(args.input, args.output)
    elif args.subcommand == "split":
        split(args.input, args.output)
    elif args.subcommand == "runbinary":
        run_binclassifier(args.model_size)
    elif args.subcommand == "runmulti":
        run_multiclassifier(args.model_size)
    elif args.subcommand == "runtrans":
        run_transformer(args.model_size)