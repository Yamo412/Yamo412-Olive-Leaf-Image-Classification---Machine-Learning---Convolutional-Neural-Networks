# scripts/main.py
# @author: Y.S

import sys
import os
from rich.console import Console
sys.path.append('./scripts')
from scripts.menu import menu
from scripts.evaluate import evaluate_all_models, generate_classification_report_and_metrics
from scripts.train import train_all_models
from scripts.comparison import compare_models_and_generate_plots  # New import

console = Console()

def clear_outputs():
    """Clear the outputs and models directories."""
    
    # Clear outputs directory
    output_dir = './outputs'
    if os.path.exists(output_dir):
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                    console.print(f"Deleted {file_name} from outputs", style="bold red")
                except Exception as e:
                    console.print(f"Error deleting {file_name} from outputs: {str(e)}", style="bold yellow")
            elif os.path.isdir(file_path):
                console.print(f"Skipping directory {file_name} in outputs", style="bold yellow")
    else:
        console.print(f"{output_dir} does not exist.", style="bold yellow")
    
    # Clear models directory
    models_dir = './models'
    if os.path.exists(models_dir):
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if os.path.isdir(model_path):
                for file_name in os.listdir(model_path):
                    file_path = os.path.join(model_path, file_name)
                    if os.path.isfile(file_path):
                        try:
                            os.unlink(file_path)
                            console.print(f"Deleted {file_name} from {model_name} in models", style="bold red")
                        except Exception as e:
                            console.print(f"Error deleting {file_name} from {model_name} in models: {str(e)}", style="bold yellow")
            else:
                console.print(f"Skipping non-directory item {model_name} in models", style="bold yellow")
    else:
        console.print(f"{models_dir} does not exist.", style="bold yellow")

def main():
    """Main loop for the CLI application."""
    while True:
        choice = menu()

        if choice == '1':
            console.print("🚀 Starting model training...")
            train_all_models()  # Train all models
        elif choice == '2':
            console.print("🧪 Evaluating models on the test set...")
            evaluate_all_models()  # Evaluate models on test set
        elif choice == '3':
            console.print("📈 Generating classification metrics for a model...")
            model_name = input("Enter the model name (e.g., InceptionV3, MobileNetV2, DenseNet121): ")
            generate_classification_report_and_metrics(model_name)  # Generate classification metrics
        elif choice == '4':
            console.print("📊 Comparing models and generating plots...")
            compare_models_and_generate_plots()  # Compare models and generate plots
        elif choice == '5':
            console.print("🗑️ Clearing output and models directories...")
            clear_outputs()  # Clear the outputs and models directory
        elif choice == '6':
            console.print("[bold green]Exiting the program...[/bold green]")
            break
        else:
            console.print("[bold red]Invalid input! Please enter a number between 1 and 6.[/bold red]")

if __name__ == '__main__':
    main()
