import os
import sys
import numpy as np
import pandas as pd
from rich.console import Console
from PIL import Image, ImageTk, ExifTags
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from evaluate import evaluate_all_models, generate_classification_report_and_metrics
from train import train_all_models
from comparison import compare_models_and_generate_plots
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import threading

# Set up console for basic output
console = Console()

IMG_HEIGHT = 224
IMG_WIDTH = 224


class OliveLeafClassifierGUI:
    def __init__(self, root):
        """Initialize the GUI components."""
        self.root = root
        self.root.title("Olive Leaf Disease Classifier")
        self.root.geometry("800x600")  # Set the window size
        self.root.configure(bg="#f0f0f0")  # Light background color
        self.root.attributes("-alpha", 0.97)  # Slight transparency (97%)

        # Declare result_label and image_path at the class level so it's accessible in all methods
        self.result_label = None
        self.image_path = None

        # Create a frame for buttons
        self.button_frame = tk.Frame(root, bg="#ffffff", relief=tk.SUNKEN, bd=2)
        self.button_frame.pack(pady=20)

        # Add buttons for menu options
        self.add_menu_buttons()

        # Version and Author labels with custom styling
        self.version_label = tk.Label(self.root, text="Version: Alpha 1.0", bg="#f0f0f0", font=("Helvetica", 9, "italic"), fg="green")
        self.version_label.place(relx=0.01, rely=0.95)
        self.author_label = tk.Label(self.root, text="@Author: Y.S", bg="#f0f0f0", font=("Helvetica", 9, "italic"), fg="green")
        self.author_label.place(relx=0.9, rely=0.95)

    def add_menu_buttons(self):
        """Add buttons for each of the options."""
        button_styles = {
            "bg": "#3498db",  # Button background color
            "fg": "#ffffff",  # Button text color
            "activebackground": "#2980b9",  # Background color when hovered
            "font": ("Helvetica", 12, "bold"),
            "width": 30
        }
        button_labels = [
            ("Train All Models", self.train_models),
            ("Evaluate All Models", self.evaluate_models),
            ("Generate Classification Metrics", self.generate_metrics),
            ("Compare Models & Generate Plots", self.compare_models),
            ("Olive Leaf Classifier", self.open_classifier_window),
            ("Clear Outputs Directory", self.clear_outputs),
            ("Exit", self.exit_program)
        ]

        for idx, (text, command) in enumerate(button_labels):
            button = tk.Button(self.button_frame, text=text, command=command, **button_styles)
            button.grid(row=idx, column=0, padx=10, pady=10)

    def setup_log_window(self, title):
        """Create a separate window for displaying logs for each task."""
        log_window = tk.Toplevel(self.root)
        log_window.title(f"Console Logs - {title}")
        log_window.geometry("800x400")
        log_text = scrolledtext.ScrolledText(log_window, wrap=tk.WORD, height=20, bg="#1e1e1e", fg="#ffffff", font=("Consolas", 10))
        log_text.pack(expand=True, fill=tk.BOTH)
        return log_text

    def get_model_names(self):
        """Automatically fetch all model names from the './models/' directory."""
        models_dir = './models'
        if os.path.exists(models_dir):
            return [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]
        else:
            console.print(f"[bold red]{models_dir} does not exist. No models available!")
            return []

    def train_models(self):
        """Run model training in a separate thread and log to a dedicated console."""
        log_text = self.setup_log_window("Train Models")
        self.run_task("🚀 Starting model training...", train_all_models, log_text)

    def evaluate_models(self):
        """Run model evaluation and log to a dedicated console."""
        log_text = self.setup_log_window("Evaluate Models")
        self.run_task("🧪 Evaluating models...", self.evaluate_and_display_table, log_text)
        
    def generate_metrics(self):
        """Generate classification metrics for all available models in a separate thread."""
        log_text = self.setup_log_window("Generate Metrics")
        model_names = self.get_model_names()

        # Run the entire loop in a separate thread
        threading.Thread(target=self.run_metrics_sequentially, args=(model_names, log_text)).start()

    def run_metrics_sequentially(self, model_names, log_text):
        """Run metrics generation for each model sequentially in a single thread."""
        for model_name in model_names:
            console.print(f"📈 Generating classification metrics for {model_name}...")
            log_text.insert(tk.END, f"📈 Generating classification metrics for {model_name}...\n")
            self.generate_metrics_to_file(model_name)

    def generate_metrics_to_file(self, model_name):
        """Generate classification metrics and save them in separate files for each model."""
        generate_classification_report_and_metrics(model_name)

    def compare_models(self):
        """Compare models and generate plots in a separate thread."""
        log_text = self.setup_log_window("Compare Models")
        self.run_task("📊 Comparing models and generating plots...", compare_models_and_generate_plots, log_text)

    def clear_outputs(self):
        """Clear the output directory in a separate thread and log to a dedicated console."""
        log_text = self.setup_log_window("Clear Outputs")
        self.run_task("🗑️ Clearing output directory...", self.clear_output_files, log_text)

    def clear_output_files(self):
        """Helper function to delete files in the outputs directory."""
        output_dir = './outputs'
        if os.path.exists(output_dir):
            for file_name in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    console.print(f"Deleted {file_name}")
        else:
            console.print(f"{output_dir} does not exist.")

    def exit_program(self):
        """Exit the application."""
        console.print("Exiting the program...")
        self.root.quit()

    def run_task(self, task_message, task_function, log_text):
        """Display a message and run a task sequentially."""
        log_text.insert(tk.END, task_message + "\n")
        console.print(task_message)
        task_function()  # Sequential execution, no threading

    def open_classifier_window(self):
        """Open the Olive Leaf Classifier window."""
        self.start_classifier_window()

    def start_classifier_window(self):
        """Create a new window for testing image classification."""
        classifier_window = tk.Toplevel(self.root)
        classifier_window.title("Olive Leaf Classifier")
        classifier_window.geometry("800x600")
        classifier_window.configure(bg="white")

        # Model selection label
        label_model = tk.Label(classifier_window, text="Select Model:", bg="white", font=("Arial", 12))
        label_model.pack(pady=10)

        # Fetch models directly from the available model directories
        model_names = self.get_model_names()
        model_var = tk.StringVar(classifier_window)
        model_var.set(model_names[0] if model_names else "No models available")

        # Display the model names in a dropdown
        model_menu = ttk.Combobox(classifier_window, textvariable=model_var, values=model_names, state="readonly")
        model_menu.pack(pady=5)

        # Create a frame for the image display area
        image_frame = tk.Frame(classifier_window, width=300, height=300, bg="lightgray", relief="solid", bd=1)
        image_frame.pack(pady=20)

        # Placeholder image label
        image_label = tk.Label(image_frame, bg="lightgray")
        image_label.place(relx=0.5, rely=0.5, anchor="center")

        # Load Image button
        button_load = tk.Button(classifier_window, text="Load Image", command=lambda: self.load_image(image_label))
        button_load.pack(pady=10)

        # Predict button
        button_predict = tk.Button(classifier_window, text="Predict", command=lambda: self.predict_image(model_var.get()))
        button_predict.pack(pady=10)

        # Prediction result label
        self.result_label = tk.Label(classifier_window, text="Prediction result will be shown here", bg="white", font=("Arial", 12))
        self.result_label.pack(pady=20)

    def load_image(self, image_label):
        """Load an image from file and display it in the classifier window."""
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = self.rotate_image_upright(image)
            image = image.resize((300, 300), Image.LANCZOS)
            image = ImageTk.PhotoImage(image)
            image_label.config(image=image)
            image_label.image = image  # Keep a reference to avoid garbage collection

    def predict_image(self, model_name):
        """Predict the classification of the loaded image using the selected model."""
        if not self.image_path:
            console.print("No image loaded! Please load an image first.")
            return

        model_path = f"./models/{model_name}/{model_name}_fold_1.keras"
        model = load_model(model_path)

        # Preprocess the image
        img = Image.open(self.image_path)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        prediction = model.predict(img_array)
        result = "Sick" if prediction >= 0.5 else "Healthy"
        confidence = prediction[0][0] * 100 if prediction >= 0.5 else (1 - prediction[0][0]) * 100

        # Display the result
        self.result_label.config(text=f"Prediction: {result} (Confidence: {confidence:.2f}%)")

    def rotate_image_upright(self, image):
        """Rotate the image upright based on EXIF data."""
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()
            if exif is not None:
                orientation = exif[orientation]
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass  # No EXIF data or other error
        return image

    def evaluate_and_display_table(self):
        """Evaluate all models, display metrics as a table, and save results to a text file."""
        model_names = self.get_model_names()
        results = []
    
        # Prepare to store the output as a string for the text file
        output = []

        for model_name in model_names:
            # Run evaluation and gather metrics for each model
            metrics = evaluate_all_models()
            metrics.insert(0, model_name)  # Insert the model name as the first column
            results.append(metrics)

        # Convert results to a Pandas DataFrame for cleaner display
        df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"])
    
        # Print to console for GUI feedback
        print(df)

        # Add the DataFrame to the output string for saving
        output.append(df.to_string(index=False))

        # Option to save as a plot
        self.plot_metrics(df)

        # Save the formatted table to a text file
        output_file_path = "./outputs/evaluation_metrics_summary.txt"
        with open(output_file_path, "w") as file:
            file.write("\n".join(output))

        # Print confirmation to console
        console.print(f"Model evaluation results saved to: {output_file_path}")


    def plot_metrics(self, df):
        """Plot the evaluation metrics as a bar plot."""
        df.plot(kind='bar', figsize=(10, 6))
        plt.title("Model Performance Metrics")
        plt.xlabel("Model")
        plt.ylabel("Metrics")
        plt.xticks(range(len(df["Model"])), df["Model"])
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


# Main function to launch the GUI
if __name__ == '__main__':
    root = tk.Tk()
    app = OliveLeafClassifierGUI(root)
    root.mainloop()
