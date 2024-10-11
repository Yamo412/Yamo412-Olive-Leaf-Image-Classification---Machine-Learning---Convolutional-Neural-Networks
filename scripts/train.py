# scripts/train.py
# @author: Y.S

# Manipulation
import pandas as pd
import sys
import os
import numpy as np
import threading

# Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

# Misc.
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.theme import Theme

# Scripts
sys.path.append('./scripts')
from data_loader import TRAIN_DIR, VALIDATION_DIR, load_data
from model_builder import build_model, fine_tune_model

# Model Names
MODEL_NAMES = ['MobileNetV2', 'InceptionV3', 'DenseNet121']

# Training parameters
EPOCHS = 25
PATIENCE = 5
N_SPLITS = 5  # For stratified K-fold cross-validation

# Set up console for styled output
theme = Theme({
    "info": "cyan",
    "warning": "bold red",
    "success": "bold green",
    "progress": "bold blue"
})

console = Console(theme=theme)

# Prepare a data generator with augmentation for cross-validation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Balanced augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load file paths and class labels manually
def load_data_paths_labels():
    """Load the file paths and labels for StratifiedKFold."""
    file_paths = []
    labels = []
    
    for class_name in os.listdir(TRAIN_DIR):
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.isdir(class_dir):
            class_label = class_name  # Use class names like 'Healthy', 'Sick'
            for fname in os.listdir(class_dir):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_label)
    
    return np.array(file_paths), np.array(labels)

def get_class_weights(train_generator):
    """Compute class weights to handle class imbalance."""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    return dict(enumerate(class_weights))

def save_training_history(history, model_name, fold):
    """Save training history as both .npy and .txt files."""
    
    # Save history as .npy
    history_path = f'./outputs/training_history_{model_name}_fold_{fold}.npy'
    np.save(history_path, history.history)
    
    # Save a human-readable .txt file
    history_txt_path = f'./outputs/training_history_{model_name}_fold_{fold}.txt'
    with open(history_txt_path, 'w') as f:
        f.write(f"Training History for {model_name} - Fold {fold}\n")
        f.write("=" * 50 + "\n")
        for key, values in history.history.items():
            f.write(f"{key}: {values}\n")
        f.write("\n")
    console.print(f"[green]Training history for {model_name} (fold {fold}) saved.[/green]")

def stratified_kfold_training(model_name):
    """Perform Stratified K-Fold Cross-Validation for a given architecture."""
    
    console.print(f"🚀 [progress]Starting stratified K-fold cross-validation for model: [bold]{model_name}[/bold]", style="info")

    # Build and compile the model
    model = build_model(model_name)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Create a folder for this model inside the 'models' directory
    model_folder = f'./models/{model_name}'
    os.makedirs(model_folder, exist_ok=True)  # Create the directory if it doesn't exist

    # Load file paths and labels
    X, y = load_data_paths_labels()

    # Initialize Stratified KFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    best_fold = None
    best_val_accuracy = -1
    fold_metrics = []

    # K-fold training
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        console.print(f"Training fold {fold}/{N_SPLITS}", style="info")

        # Prepare training and validation data generators
        train_generator = datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({"filename": X[train_idx], "class": y[train_idx]}),
            x_col="filename",
            y_col="class",
            target_size=(224, 224),
            batch_size=32,
            class_mode="binary"
        )
        
        val_generator = datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({"filename": X[val_idx], "class": y[val_idx]}),
            x_col="filename",
            y_col="class",
            target_size=(224, 224),
            batch_size=32,
            class_mode="binary"
        )

        # Get class weights to address class imbalance
        class_weights = get_class_weights(train_generator)

        # Progress bar for epochs
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(f"[cyan]Training fold {fold}/{N_SPLITS} for model {model_name}...", total=EPOCHS)

            # Checkpoint to save the best model for each fold inside its own model folder
            checkpoint = ModelCheckpoint(
                filepath=os.path.join(model_folder, f'{model_name}_fold_{fold}.keras'),  # Save each fold separately
                save_best_only=True, 
                monitor='val_loss', 
                mode='min',
                verbose=1
            )

            # Train for all epochs with class weights
            history = model.fit(
                train_generator,
                epochs=EPOCHS,
                validation_data=val_generator,
                class_weight=class_weights,  # Apply class weights here
                callbacks=[early_stopping, checkpoint, lr_scheduler],  # Apply learning rate scheduler
                verbose=1
            )

            # Save training history
            save_training_history(history, model_name, fold)

            # Fine-tuning: Unfreeze base model and re-train
            model = fine_tune_model(model, model_name)
            history_fine_tuning = model.fit(
                train_generator,
                epochs=EPOCHS // 2,  # Fewer epochs for fine-tuning
                validation_data=val_generator,
                class_weight=class_weights,
                callbacks=[early_stopping, checkpoint, lr_scheduler],
                verbose=1
            )

            # Save training history for fine-tuning as well
            save_training_history(history_fine_tuning, model_name, f'{fold}_fine_tuning')

            # Calculate validation accuracy for this fold
            val_accuracy = max(history.history['val_accuracy'])
            fold_metrics.append((fold, val_accuracy))
            
            # Save the best fold model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_fold = fold

            # Update progress bar with completed epochs
            progress.update(task, completed=history.epoch[-1] + 1)

        console.print(f"[success]Training completed for fold {fold}/{N_SPLITS} for model: {model_name}[/success]")

    console.print(f"\n[green]Best fold: {best_fold} with validation accuracy: {best_val_accuracy:.4f}[/green]")

def train_all_models():
    """Train all models sequentially using stratified K-fold cross-validation."""
    for model_name in MODEL_NAMES:
        stratified_kfold_training(model_name)

if __name__ == '__main__':
    console.print("[bold cyan]Training Models using Stratified K-Fold Cross-Validation...[/bold cyan]")
    train_all_models()
