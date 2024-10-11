# scripts/comparison.py
# @author: Y.S

import matplotlib.pyplot as plt
import numpy as np
import os
from rich.console import Console

# Models to compare
MODEL_NAMES = ['MobileNetV2', 'InceptionV3', 'DenseNet121']
FOLD_COUNT = 5  # Update this based on the number of folds
USE_FINE_TUNING = True  # Set this to True if you want to include fine-tuning

# Set up a console for styled output
console = Console()

def load_training_history(model_name):
    """Load and combine the saved training histories for all folds and optionally fine-tuning."""
    combined_history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }

    for fold in range(1, FOLD_COUNT + 1):
        # Load standard training history
        history_path = f'./outputs/training_history_{model_name}_fold_{fold}.npy'
        if os.path.exists(history_path):
            console.print(f"[bold green]Loading history for {model_name}, fold {fold}...[/bold green]")
            history = np.load(history_path, allow_pickle=True).item()
            for key in combined_history:
                combined_history[key].extend(history.get(key, []))  # Extend to combine epochs across folds

        # Optionally load fine-tuning history
        if USE_FINE_TUNING:
            fine_tuning_path = f'./outputs/training_history_{model_name}_fold_{fold}_fine_tuning.npy'
            if os.path.exists(fine_tuning_path):
                console.print(f"[bold green]Loading fine-tuning history for {model_name}, fold {fold}...[/bold green]")
                fine_tuning_history = np.load(fine_tuning_path, allow_pickle=True).item()
                for key in combined_history:
                    combined_history[key].extend(fine_tuning_history.get(key, []))

    return combined_history

def plot_learning_curves(histories, model_names):
    """Plot learning curves (training vs validation accuracy) for each model."""
    plt.figure(figsize=(10, 6))
    for history, model_name in zip(histories, model_names):
        if history is not None and len(history['accuracy']) > 0:
            plt.plot(history['accuracy'], label=f'{model_name} - Training Accuracy')
            plt.plot(history['val_accuracy'], label=f'{model_name} - Validation Accuracy')

    plt.title('Learning Curves: Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('./outputs/learning_curves_comparison.png')
    plt.show()

def plot_validation_accuracy(histories, model_names):
    """Plot validation accuracy as a function of epoch number."""
    plt.figure(figsize=(10, 6))
    for history, model_name in zip(histories, model_names):
        if history is not None and len(history['val_accuracy']) > 0:
            plt.plot(history['val_accuracy'], label=f'{model_name} - Validation Accuracy')

    plt.title('Validation Accuracy as a Function of Epoch Number')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('./outputs/validation_accuracy_comparison.png')
    plt.show()

def plot_training_accuracy(histories, model_names):
    """Plot training accuracy comparison across models."""
    plt.figure(figsize=(10, 6))
    for history, model_name in zip(histories, model_names):
        if history is not None and len(history['accuracy']) > 0:
            plt.plot(history['accuracy'], label=f'{model_name} - Training Accuracy')

    plt.title('Training Accuracy Comparison Across Models')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('./outputs/training_accuracy_comparison.png')
    plt.show()

def compare_models_and_generate_plots():
    """Load training histories for all models and generate comparison plots."""
    histories = [load_training_history(model_name) for model_name in MODEL_NAMES]

    # Generate all plots
    plot_learning_curves(histories, MODEL_NAMES)
    plot_validation_accuracy(histories, MODEL_NAMES)
    plot_training_accuracy(histories, MODEL_NAMES)

if __name__ == '__main__':
    compare_models_and_generate_plots()
