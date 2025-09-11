import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import logging
from model.model_multiclass import load_model_multiclass, NetworkTrafficDatasetMulticlass
from trainer.trainer_multiclass import evaluate_model_multiclass
from sklearn.metrics import precision_recall_fscore_support
from model.model_transformer import load_model_transformer, NetworkTrafficDatasetTransformer
from trainer.trainer_transformer import evaluate_model_transformer

logger = logging.getLogger(__name__)

# # # # MLP

def plot_test_set_per_class_metrics(predictions, targets, class_names, save_path=None):
   
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = np.arange(len(class_names))
    
    colors = cm.Set3(np.linspace(0, 1, len(class_names)))
    
    x_pos_agg = np.arange(len(class_names) + 1)
    f1_with_agg = np.append(f1_per_class, f1_weighted)
    class_names_with_agg = list(class_names) + ['Aggregated']
    
    colors_agg = np.vstack([colors, [0.5, 0.5, 0.5, 1.0]])
    
    bars1 = ax1.bar(x_pos_agg, f1_with_agg, color=colors_agg, alpha=1.0)
    ax1.set_title('F1-Score per Classe + Aggregated', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classi')
    ax1.set_ylabel('F1-Score')
    ax1.set_xticks(x_pos_agg)
    ax1.set_xticklabels(class_names_with_agg, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
     
    bars2 = ax2.bar(x_pos, precision_per_class, color=colors, alpha=1.0)
    ax2.set_title('Precision per Classe', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Classi')
    ax2.set_ylabel('Precision')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    bars3 = ax3.bar(x_pos, recall_per_class, color=colors, alpha=1.0)
    ax3.set_title('Recall per Classe', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Classi')
    ax3.set_ylabel('Recall')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    bars4 = ax4.bar(x_pos, support_per_class, color=colors, alpha=1.0)
    ax4.set_title('Support per Classe - Numero Campioni', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Classi')
    ax4.set_ylabel('Numero Campioni')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    max_support = max(support_per_class)
    ax4.set_ylim(0, max_support)
    
    accuracy = (predictions == targets).mean()
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        targets, predictions, average='macro', zero_division=0
    )
    
    
    plt.tight_layout()
    
    if save_path:
        test_metrics_path = save_path.replace('.png', '_test_set_per_class_metrics.png')
        plt.savefig(test_metrics_path, dpi=300, bbox_inches='tight')
        logger.info(f"Test set per-class metrics plot salvato in: {test_metrics_path}")
    
    plt.show()
    
    if save_path:
        save_individual_test_plots(predictions, targets, class_names, save_path)
    
    logger.info("\n" + "="*60)
    logger.info("METRICHE PER CLASSE")
    logger.info("="*60)
    
    class_metrics_data = []
    for i, class_name in enumerate(class_names):
        class_metrics_data.append({
            'name': class_name,
            'f1': f1_per_class[i],
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'support': support_per_class[i]
        })
    
    class_metrics_data.append({
        'name': 'AGGREGATED (Weighted)',
        'f1': f1_weighted,
        'precision': precision_w,
        'recall': recall_w,
        'support': len(targets)
    })
    
    class_metrics_individual = [x for x in class_metrics_data if 'AGGREGATED' not in x['name']]
    class_metrics_individual.sort(key=lambda x: x['f1'])
    class_metrics_aggregated = [x for x in class_metrics_data if 'AGGREGATED' in x['name']]
    class_metrics_data = class_metrics_individual + class_metrics_aggregated
    
    for cls_data in class_metrics_data:
        if cls_data['f1'] < 0.3:
            status = "🔴"
        elif cls_data['f1'] < 0.5:
            status = "🟡"
        elif cls_data['f1'] < 0.8:
            status = "🟢"
        else:
            status = "✅"
        
        if 'AGGREGATED' in cls_data['name']:
            status = "📊"
            logger.info("-" * 60)
        
        logger.info(f"{status} {cls_data['name']:20s}: "
                f"F1={cls_data['f1']:.4f} | "
                f"P={cls_data['precision']:.4f} | "
                f"R={cls_data['recall']:.4f} | "
                f"Support={int(cls_data['support']):5d}")
    
    individual_f1s = [cls['f1'] for cls in class_metrics_individual]
    avg_f1 = np.mean(individual_f1s)
    worst_f1 = np.min(individual_f1s)
    best_f1 = np.max(individual_f1s)
    critical_classes = len([f1 for f1 in individual_f1s if f1 < 0.3])
    
    logger.info("\n" + "-"*40)
    logger.info(f"Media F1 (classi individuali): {avg_f1:.4f}")
    logger.info(f"Migliore F1: {best_f1:.4f}")
    logger.info(f"Peggiore F1: {worst_f1:.4f}")
    logger.info(f"Classi critiche (F1<0.3): {critical_classes}")
    logger.info("-"*40)

def save_individual_test_plots(predictions, targets, class_names, base_save_path):
    
    # Calcola metriche
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    
    # Setup colori e posizioni
    x_pos = np.arange(len(class_names))
    colors = cm.Set3(np.linspace(0, 1, len(class_names)))
    
    # 1. F1 Score per classe + Aggregated
    plt.figure(figsize=(12, 8))
    x_pos_agg = np.arange(len(class_names) + 1)
    f1_with_agg = np.append(f1_per_class, f1_weighted)
    class_names_with_agg = list(class_names) + ['Aggregated']
    
    colors_agg = np.vstack([colors, [0.5, 0.5, 0.5, 1.0]])
    
    bars = plt.bar(x_pos_agg, f1_with_agg, color=colors_agg, alpha=1.0)
    plt.title('F1-Score per Classe + Aggregated', fontsize=14, fontweight='bold')
    plt.xlabel('Classi')
    plt.ylabel('F1-Score')
    plt.xticks(x_pos_agg, class_names_with_agg, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    f1_path = base_save_path.replace('.png', '_test_f1_scores.png')
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Test F1 scores plot salvato in: {f1_path}")
    
    # 2-4. Altri grafici (Precision, Recall, Support)
    metrics = [
        ('Precision', precision_per_class, '_test_precision.png'),
        ('Recall', recall_per_class, '_test_recall.png'),
        ('Support', support_per_class, '_test_support.png')
    ]
    
    for metric_name, metric_values, filename_suffix in metrics:
        plt.figure(figsize=(12, 8))
        bars = plt.bar(x_pos, metric_values, color=colors, alpha=1.0)
        plt.title(f'{metric_name} per Classe ', fontsize=14, fontweight='bold')
        plt.xlabel('Classi')
        plt.ylabel(metric_name)
        plt.xticks(x_pos, class_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        if metric_name != 'Support':
            plt.ylim(0, 1)
        else:
            plt.ylim(0, max(metric_values))
        
        plt.tight_layout()
        save_path = base_save_path.replace('.png', filename_suffix)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Test {metric_name} plot salvato in: {save_path}")

def test_and_plot_mlp(mlp_path):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, hyperparams, feature_columns, class_mapping = load_model_multiclass(
        mlp_path, device
    )
    
    # 2. Prepara dataset
    dataset_manager = NetworkTrafficDatasetMulticlass(
        model_size="small",
        metadata_path="resources/datasets/mlp_multiclass_metadata.json"
    )
    
    dataset_manager.load_data(
        train_path="resources/datasets/train_multiclass.csv",
        val_path="resources/datasets/val_multiclass.csv",
        test_path="resources/datasets/test_multiclass.csv"
    )
    
    _, _, test_loader = dataset_manager.create_dataloaders()
    
    class_names = dataset_manager.multiclass_metadata['label_encoder_classes']
    accuracy, precision, recall, f1, predictions, targets, probabilities = evaluate_model_multiclass(
        model, test_loader, device, class_names
    )
    
    import os
    os.makedirs("plots", exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_name = "TON"
    if (mlp_path=="models/CIC_best_mlp_multiclass_20250911_161451.pth"):
        dataset_name="CIC"
    elif (mlp_path=="models/NB_best_mlp_multiclass_20250911_120415.pth"):
        dataset_name="NB15"
    
    plot_test_set_per_class_metrics(
        predictions, 
        targets, 
        class_names, 
        save_path=f'plots/{dataset_name}_mlp_test_set_metrics_{timestamp}.png'
    )
    
    return accuracy, precision, recall, f1

# # # # TRANSFORMER

def plot_transformer_test_set_per_class_metrics(predictions, targets, class_names, save_path=None):

    
    # Calcola metriche per classe
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    
    # Calcola F1 weighted per "aggregated"
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    
    # Crea figure con layout 2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = np.arange(len(class_names))
    
    # F1 Score per classe con colori diversi + aggregated
    colors = cm.Set3(np.linspace(0, 1, len(class_names)))
    
    # Aggiunge F1 weighted per "aggregated"
    x_pos_agg = np.arange(len(class_names) + 1)
    f1_with_agg = np.append(f1_per_class, f1_weighted)
    class_names_with_agg = list(class_names) + ['Aggregated']
    
    # Colori: classi normali + grigio per aggregated
    colors_agg = np.vstack([colors, [0.5, 0.5, 0.5, 1.0]])
    
    bars1 = ax1.bar(x_pos_agg, f1_with_agg, color=colors_agg, alpha=1.0)
    ax1.set_title('F1-Score per Classe + Aggregated (Transformer)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Classi')
    ax1.set_ylabel('F1-Score')
    ax1.set_xticks(x_pos_agg)
    ax1.set_xticklabels(class_names_with_agg, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
     
    # Precision per classe (colori diversi)
    bars2 = ax2.bar(x_pos, precision_per_class, color=colors, alpha=1.0)
    ax2.set_title('Precision per Classe (Transformer)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Classi')
    ax2.set_ylabel('Precision')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Recall per classe (colori diversi)
    bars3 = ax3.bar(x_pos, recall_per_class, color=colors, alpha=1.0)
    ax3.set_title('Recall per Classe (Transformer)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Classi')
    ax3.set_ylabel('Recall')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Support (numero sequenze) per classe (colori diversi)
    bars4 = ax4.bar(x_pos, support_per_class, color=colors, alpha=1.0)
    ax4.set_title('Support per Classe - Numero Sequenze (Transformer)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Classi')
    ax4.set_ylabel('Numero Sequenze')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Per support, usa il valore massimo come riferimento
    max_support = max(support_per_class)
    ax4.set_ylim(0, max_support)
    
    # Aggiungi metriche generali come testo
    accuracy = (predictions == targets).mean()
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        targets, predictions, average='macro', zero_division=0
    )
    
    plt.tight_layout()
    
    if save_path:
        test_metrics_path = save_path.replace('.png', '_transformer_test_set_per_class_metrics.png')
        plt.savefig(test_metrics_path, dpi=300, bbox_inches='tight')
        logger.info(f"Transformer test set per-class metrics plot salvato in: {test_metrics_path}")
    
    plt.show()
    
    # Salva anche i grafici individuali per il test set
    if save_path:
        save_individual_transformer_test_plots(predictions, targets, class_names, save_path)
    
    # Log delle metriche (identico ma con titolo Transformer)
    logger.info("\n" + "="*70)
    logger.info("METRICHE PER CLASSE TRANSFORMER (TEST SET)")
    logger.info("="*70)
    
    # Crea tabella ordinata per F1 (identico al codice originale)
    class_metrics_data = []
    for i, class_name in enumerate(class_names):
        class_metrics_data.append({
            'name': class_name,
            'f1': f1_per_class[i],
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'support': support_per_class[i]
        })
    
    # Aggiungi metriche aggregate
    class_metrics_data.append({
        'name': 'AGGREGATED (Weighted)',
        'f1': f1_weighted,
        'precision': precision_w,
        'recall': recall_w,
        'support': len(targets)
    })
    
    # Ordina per F1 (peggiori prima, ma aggregated sempre ultimo)
    class_metrics_individual = [x for x in class_metrics_data if 'AGGREGATED' not in x['name']]
    class_metrics_individual.sort(key=lambda x: x['f1'])
    class_metrics_aggregated = [x for x in class_metrics_data if 'AGGREGATED' in x['name']]
    class_metrics_data = class_metrics_individual + class_metrics_aggregated
    
    for cls_data in class_metrics_data:
        # Emoji basato su F1 score
        if cls_data['f1'] < 0.3:
            status = "🔴"
        elif cls_data['f1'] < 0.5:
            status = "🟡"
        elif cls_data['f1'] < 0.8:
            status = "🟢"
        else:
            status = "✅"
        
        # Formatting speciale per aggregated
        if 'AGGREGATED' in cls_data['name']:
            status = "📊"
            logger.info("-" * 70)
        
        logger.info(f"{status} {cls_data['name']:20s}: "
                f"F1={cls_data['f1']:.4f} | "
                f"P={cls_data['precision']:.4f} | "
                f"R={cls_data['recall']:.4f} | "
                f"Seq={int(cls_data['support']):5d}")
    
    # Statistiche generali
    individual_f1s = [cls['f1'] for cls in class_metrics_individual]
    avg_f1 = np.mean(individual_f1s)
    worst_f1 = np.min(individual_f1s)
    best_f1 = np.max(individual_f1s)
    critical_classes = len([f1 for f1 in individual_f1s if f1 < 0.3])
    
    logger.info("\n" + "-"*50)
    logger.info(f"Media F1 (classi individuali): {avg_f1:.4f}")
    logger.info(f"Migliore F1: {best_f1:.4f}")
    logger.info(f"Peggiore F1: {worst_f1:.4f}")
    logger.info(f"Classi critiche (F1<0.3): {critical_classes}")
    logger.info("-"*50)


def save_individual_transformer_test_plots(predictions, targets, class_names, base_save_path):
    
    # Calcola metriche
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    
    # Setup colori e posizioni
    x_pos = np.arange(len(class_names))
    colors = cm.Set3(np.linspace(0, 1, len(class_names)))
    
    # 1. F1 Score per classe + Aggregated
    plt.figure(figsize=(12, 8))
    x_pos_agg = np.arange(len(class_names) + 1)
    f1_with_agg = np.append(f1_per_class, f1_weighted)
    class_names_with_agg = list(class_names) + ['Aggregated']
    
    colors_agg = np.vstack([colors, [0.5, 0.5, 0.5, 1.0]])
    
    bars = plt.bar(x_pos_agg, f1_with_agg, color=colors_agg, alpha=1.0)
    plt.title('F1-Score per Classe + Aggregated (Transformer)', fontsize=14, fontweight='bold')
    plt.xlabel('Classi')
    plt.ylabel('F1-Score')
    plt.xticks(x_pos_agg, class_names_with_agg, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    f1_path = base_save_path.replace('.png', '_transformer_test_f1_scores.png')
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Transformer Test F1 scores plot salvato in: {f1_path}")
    
    # 2-4. Altri grafici (Precision, Recall, Support)
    metrics = [
        ('Precision', precision_per_class, '_transformer_test_precision.png'),
        ('Recall', recall_per_class, '_transformer_test_recall.png'),
        ('Support', support_per_class, '_transformer_test_support.png')
    ]
    
    for metric_name, metric_values, filename_suffix in metrics:
        plt.figure(figsize=(12, 8))
        bars = plt.bar(x_pos, metric_values, color=colors, alpha=1.0)
        plt.title(f'{metric_name} per Classe (Transformer)', fontsize=14, fontweight='bold')
        plt.xlabel('Classi')
        plt.ylabel(metric_name if metric_name != 'Support' else 'Numero Sequenze')
        plt.xticks(x_pos, class_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        if metric_name != 'Support':
            plt.ylim(0, 1)
        else:
            plt.ylim(0, max(metric_values))
        
        plt.tight_layout()
        save_path = base_save_path.replace('.png', filename_suffix)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Transformer Test {metric_name} plot salvato in: {save_path}")


def test_and_plot_transformer(trans_path, window_size=8):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Carica modello Transformer
    model, _ = load_model_transformer(
        trans_path, device
    )
    
    # 2. Prepara dataset Transformer
    dataset_manager = NetworkTrafficDatasetTransformer(
        model_size="small",  # O il size usato durante il training
        metadata_path="resources/datasets/transformer_metadata.json"
    )
    
    feature_dim = dataset_manager.load_processed_data(
        train_csv_path="resources/datasets/train_transformer_processed.csv",
        val_csv_path="resources/datasets/val_transformer_processed.csv",
        test_csv_path="resources/datasets/test_transformer_processed.csv"
    )
    
    # Crea dataloaders con la stessa window_size del training
    _, _, test_loader = dataset_manager.create_dataloaders(
        window_size=window_size, seed=42
    )
    
    # 3. Testa modello per ottenere predizioni
    class_names = dataset_manager.temporal_metadata['label_encoder_classes']
    accuracy, precision, recall, f1, predictions, targets, probabilities = evaluate_model_transformer(
        model, test_loader, device, class_names, dataset_manager
    )
    
    import os
    os.makedirs("plots", exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dataset_name = "TON"
    if (trans_path=="models/CIC_best_transformer_20250911_150635.pth"):
        dataset_name="CIC"
    elif (trans_path=="models/NB_best_transformer_20250911_104438.pth"):
        dataset_name="NB15"

    plot_transformer_test_set_per_class_metrics(
        predictions, 
        targets, 
        class_names, 
        save_path=f'plots/{dataset_name}_transformer_test_set_metrics_{timestamp}.png'
    )
    
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    mlp_path = "models/TON_best_mlp_multiclass_20250910_234927.pth"
    test_and_plot_mlp(mlp_path)
    trans_path = "models/TON_best_transformer_20250910_191111.pth"
    test_and_plot_transformer(trans_path)