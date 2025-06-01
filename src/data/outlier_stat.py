import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_outlier_impact_table(df, cols):
    """
    Confronta outlier prima e dopo log1p per le feature fornite
    e mostra i risultati in una tabella con matplotlib.
    """
    results = []

    for col in cols:
        raw = df[col].dropna()
        logged = np.log1p(raw)

        def get_outlier_info(series):
            Q1 = np.percentile(series, 25)
            Q3 = np.percentile(series, 75)
            IQR = Q3 - Q1
            low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = ((series < low) | (series > high)).sum()
            pct = outliers / len(series) * 100
            return round(outliers, 2), round(pct, 2), round(IQR, 2)

        raw_outliers, raw_pct, raw_iqr = get_outlier_info(raw)
        log_outliers, log_pct, log_iqr = get_outlier_info(logged)

        results.append([
            col, raw_pct, log_pct, raw_iqr, log_iqr
        ])

    # Crea DataFrame per facilità di lettura
    result_df = pd.DataFrame(results, columns=[
        "Feature", "Outliers Prima (%)", "Outliers Dopo log1p (%)", "IQR Prima", "IQR Dopo"
    ])

    # Plot tabella
    fig, ax = plt.subplots(figsize=(10, len(cols) * 0.5 + 2))
    ax.axis('off')
    table = ax.table(
        cellText=result_df.values,
        colLabels=result_df.columns,
        cellLoc='center',
        loc='center'
    )

    table.scale(1.2, 1.2)
    plt.title("Impatto del log1p sugli Outlier e IQR", fontsize=14, pad=20)
    plt.show()

    return result_df
