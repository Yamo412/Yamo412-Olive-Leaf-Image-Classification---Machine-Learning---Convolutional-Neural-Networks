import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.models import load_model  # type: ignore

# Scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from data_loader import load_data

# Initialize a console for rich printing
console = Console()

# Models to compare
MODEL_NAMES = ['MobileNetV2', 'InceptionV3', 'DenseNet121']

# Load data
_, _, test_generator = load_data()

# Helper function to clean the ANSI escape sequences
def remove_ansi_escape_sequences(text):
    """Removes ANSI escape sequences from the text."""
    import re
    ansi_escape = re.compile(r'(?:\x1B[@-_][0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def load_training_history(model_name):
    """Load the saved training history for a given model."""
    history_path = f'./outputs/training_history_{model_name}_fold_1.npy'
    if os.path.exists(history_path):
        console.print(f"[bold green]Loading training history for {model_name}...[/bold green]")
        return np.load(history_path, allow_pickle=True).item()
    else:
        console.print(f"[bold red]No training history found for {model_name}. Please train the model first![/bold red]")
        return None

def evaluate_all_models():
    """Evaluate all models on the test set and print performance metrics."""
    # Initialize table for metrics
    table = Table(title="Model Performance Metrics on Test Set")
    table.add_column("Model", justify="center", style="cyan")
    table.add_column("Accuracy", justify="center", style="green")
    table.add_column("Precision", justify="center", style="yellow")
    table.add_column("Recall", justify="center", style="magenta")
    table.add_column("F1-Score", justify="center", style="blue")
    table.add_column("ROC AUC", justify="center", style="white")

    for model_name in MODEL_NAMES:
        console.print(f"\nEvaluating {model_name} on the test set...")

        # Load the model
        model = load_model(f'./models/{model_name}/{model_name}_fold_1.keras')

        # Load the training history
        history = load_training_history(model_name)

        if history is None:
            continue

        # Predict on the test set
        y_true = test_generator.classes
        y_pred_prob = model.predict(test_generator)
        y_pred = np.where(y_pred_prob > 0.5, 1, 0)  # Binarize predictions

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_prob)

        # Add row to the table
        table.add_row(
            model_name,
            f"{accuracy:.2f}",
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"{f1:.2f}",
            f"{roc_auc:.2f}"
        )

        # Save the cleaned output to a file
        output_text = remove_ansi_escape_sequences(table.__str__())  # Cleaning ANSI sequences
        with open(f'./outputs/{model_name}_performance.txt', 'w') as f:
            f.write(output_text)

    # Print the table to the console
    console.print(table)

def generate_classification_report_and_metrics(model_name, num_folds=5):
    """Generate classification metrics (Precision, Recall, F1-score, etc.) and report for each fold."""
    
    cumulative_metrics = {
        "accuracy": 0,
        "roc_auc": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0
    }

    for fold in range(1, num_folds + 1):
        console.print(f"Evaluating fold {fold} for {model_name}...")

        # Load the model for the current fold
        model = load_model(f'./models/{model_name}/{model_name}_fold_{fold}.keras')

        # Get the true labels and predictions
        y_true = test_generator.classes
        y_pred_prob = model.predict(test_generator)
        y_pred = np.where(y_pred_prob > 0.5, 1, 0)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_prob)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Accumulate metrics for averaging later
        cumulative_metrics["accuracy"] += accuracy
        cumulative_metrics["roc_auc"] += roc_auc
        cumulative_metrics["precision"] += precision
        cumulative_metrics["recall"] += recall
        cumulative_metrics["f1"] += f1

        # Generate and display the classification report
        report = classification_report(y_true, y_pred, target_names=['Healthy', 'Sick'])
        console.print(f"\n[bold green]Classification Report for {model_name} - Fold {fold}[/bold green]:\n")
        console.print(report)

    # Calculate average metrics across all folds
    # Calculate average metrics across all folds
    avg_metrics = {metric: cumulative_metrics[metric] / num_folds for metric in cumulative_metrics}

    # Print the average metrics
    console.print(f"\n[bold cyan]Average Classification Metrics for {model_name} ({num_folds} folds)[/bold cyan]:")
    console.print(f"Average Accuracy: {avg_metrics['accuracy']:.2f}")
    console.print(f"Average ROC AUC: {avg_metrics['roc_auc']:.2f}")
    console.print(f"Average Precision: {avg_metrics['precision']:.2f}")
    console.print(f"Average Recall: {avg_metrics['recall']:.2f}")
    console.print(f"Average F1-Score: {avg_metrics['f1']:.2f}")

    # Save the metrics summary to a file
    summary_path = f'./outputs/metrics_summary_{model_name}.txt'
    with open(summary_path, 'w') as file:
        file.write(f"Average Classification Metrics for {model_name} ({num_folds} folds):\n")
        file.write(f"Average Accuracy: {avg_metrics['accuracy']:.2f}\n")
        file.write(f"Average ROC AUC: {avg_metrics['roc_auc']:.2f}\n")
        file.write(f"Average Precision: {avg_metrics['precision']:.2f}\n")
        file.write(f"Average Recall: {avg_metrics['recall']:.2f}\n")
        file.write(f"Average F1-Score: {avg_metrics['f1']:.2f}\n")

    console.print(f"\n[bold green]Classification metrics saved to: {summary_path}[/bold green]")
