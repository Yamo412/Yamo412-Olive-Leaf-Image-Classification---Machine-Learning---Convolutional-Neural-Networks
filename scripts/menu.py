# scripts/menu.py
# @author: Y.S

from rich.console import Console

console = Console()

def menu():
    console.print("=== [bold blue]Olive Leaf Disease Classifier CLI[/bold blue] ===")
    console.print("[1] Train all models")
    console.print("[2] Evaluate models on the test set")
    console.print("[3] Generate classification metrics for a model (Precision, Recall, F1-score, ROC AUC)")
    console.print("[4] Compare models & generate plots")
    console.print("[5] Clear outputs directory")
    console.print("[6] Exit")

    choice = input("Please select an option (1-6): ")
    return choice
