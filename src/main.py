import sys
import logging

import pandas as pd
#import matplotlib

from utilities.logging_config import setup_logging
from utilities.argument_parser import ArgumentParser

from data.outlier_stat import compare_outlier_impact_table
from data.bin_preprocessing import preprocess_dataset_binary
from data.multi_preprocessing import preprocess_dataset_multiclass
from trainer.trainer_bin import main_pipeline_bin
from trainer.trainer_multiclass import main_pipeline_multiclass
import numpy as np

setup_logging()
logger = logging.getLogger(__name__)

def analyze_log1p_effect(df: pd.DataFrame, features: list[str]):
    print("=== Confronto prima e dopo log1p ===\n")

    rows = []
    for col in features:
        raw = df[col].copy()
        logged = np.log1p(raw)

        # Calcolo soglie IQR prima
        Q1_raw, Q3_raw = np.percentile(raw, [25, 75])
        IQR_raw = Q3_raw - Q1_raw
        low_raw, high_raw = Q1_raw - 1.5 * IQR_raw, Q3_raw + 1.5 * IQR_raw
        outliers_raw = ((raw < low_raw) | (raw > high_raw)).sum()
        pct_outliers_raw = outliers_raw / len(raw) * 100

        # Calcolo soglie IQR dopo log1p
        Q1_log, Q3_log = np.percentile(logged, [25, 75])
        IQR_log = Q3_log - Q1_log
        low_log, high_log = Q1_log - 1.5 * IQR_log, Q3_log + 1.5 * IQR_log
        outliers_log = ((logged < low_log) | (logged > high_log)).sum()
        pct_outliers_log = outliers_log / len(logged) * 100

        rows.append({
            "Feature": col,
            "Outliers Prima (%)": round(pct_outliers_raw, 2),
            "Outliers Dopo log1p (%)": round(pct_outliers_log, 2),
            "IQR Prima": round(IQR_raw, 2),
            "IQR Dopo": round(IQR_log, 2)
        })

    result_df = pd.DataFrame(rows)
    print(result_df.sort_values(by="Feature").to_string(index=False))

def count_outliers(df, threshold=1.5):
    """
    Conta gli outlier per ogni colonna numerica nel dataframe.
    
    Args:
        df: DataFrame - il dataframe da analizzare
        method: str - metodo per rilevare gli outlier ('iqr' o 'zscore')
        threshold: float - soglia per identificare gli outlier (1.5 per IQR, 3 per zscore)
    
    Returns:
        DataFrame con il conteggio e la percentuale di outlier per ogni feature
    """
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
    time_cols = [
        "FLOW_DURATION_MILLISECONDS",
        "DURATION_IN", "DURATION_OUT",
        "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_STDDEV",
        "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_STDDEV"
    ]
    numeric_cols = [col for col in all_cols if col not in time_cols]

    results_data = []
    
    for col in numeric_cols:
        # Conta valori NaN
        n_nan = df[col].isna().sum()
        # Ottieni i valori non-NaN
        values = df[col].dropna()
        n_total = len(df)
        n_valid = len(values)
        
        if n_valid > 0:  # Procedi solo se ci sono valori validi

            # Metodo IQR (Interquartile Range)
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = values[(values < lower_bound) | (values > upper_bound)]

            n_outliers = len(outliers)
            # Percentuale calcolata sui valori non-NaN
            perc_of_valid = (n_outliers / n_valid * 100) if n_valid > 0 else 0
            # Percentuale calcolata sul totale (inclusi NaN)
            perc_of_total = (n_outliers / n_total * 100) if n_total > 0 else 0
        else:
            # Se non ci sono valori validi
            n_outliers = 0
            perc_of_valid = 0
            perc_of_total = 0
        
        results_data.append({
            'Feature': col,
            'Totale Righe': n_total,
            'Valori NaN': n_nan,
            'Valori Validi': n_valid,
            'Numero di Outlier': n_outliers,
            'Percentuale su Validi (%)': perc_of_valid,
            'Percentuale su Totale (%)': perc_of_total
        })
    
    # crea tabella risultati
    results = pd.DataFrame(results_data)
    
    return results

def prepare_data(input_path: str, output_dir: str):
    logger.info("Preparing data...")
    df = pd.read_csv(input_path)
    
    """
    logger.info(f"Numero totale di righe: {len(df)}")
    logger.info("Risultati dell'analisi degli outlier:")
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
    table = compare_outlier_impact_table(df, all_cols)
    """
    try:
        # esecuzione del preprocessing binario
        #df_train, df_val, df_test, scaler, freq_mappings = preprocess_dataset_binary(
        #    dataset_path=input_path,
        #    config_path="config/dataset.json",
        #    output_dir=output_dir
        #)

        # esecuzione del preprocessing multiclasse
        df_train, df_val, df_test, scaler, freq_mappings, label_encoder, class_mapping = preprocess_dataset_multiclass(
            dataset_path=input_path,
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

def run_binclassifier (model_size: str):
    logger.info("Preparando il classificatore binario...")
    main_pipeline_bin()
    logger.info("✅ Training completato con successo!")

def run_multiclassifier (model_size:str):
    logger.info("Preparando il classificatore multiclasse...")
    main_pipeline_multiclass(model_size="small")
    logger.info("✅ Training completato con successo!")


if __name__ == "__main__":
    parser = ArgumentParser("parser")

    parser.register_subcommand(
        subcommand="prepare",
        arguments=["--input", "--output"],
        helps=[
            "The input path for the data.",
            "The output directory for the prepared data.",
        ],
        defaults=["resources/datasets/NF-UNSW-NB15-v3.csv", "resources/datasets"],
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
    )

    args = parser.parse_arguments(sys.argv[1:])

    if args.subcommand == "prepare":
        prepare_data(args.input, args.output)
    elif args.subcommand == "runbinary":
        run_binclassifier(args.model_size)
    elif args.subcommand == "runmulti":
        run_multiclassifier(args.model_size)