import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import LabelEncoder
import time

# Record the start time before creating the GUI window
start_gui_time = time.time()

# Load healthcare dataset
df = pd.read_csv('healthcare_dataset.csv')

# Convert categorical variables to numerical values
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Define your anonymization method (placeholder)
def anonymize_data(input_data):
    # Placeholder anonymization method
    # Replace this with your anonymization logic
    return input_data

def evaluate_anonymized_data(original_data, anonymized_data):
    # Placeholder for evaluation statistics
    pruning_power = np.random.rand()  # Placeholder for pruning power
    precision = np.random.rand()  # Placeholder for precision
    cpu_time = np.random.rand()  # Placeholder for CPU time
    communication_cost = np.random.rand()  # Placeholder for communication cost

    return pruning_power, precision, cpu_time, communication_cost

# Function to update original data frame
def update_original_data_frame():
    for i, column in enumerate(df.columns):
        tk.Label(original_data_frame_inner, text=column).grid(row=0, column=i, padx=5, pady=2)
    for i, row in enumerate(df.values[:100]):  # Limiting to 100 rows for faster rendering
        for j, value in enumerate(row):
            tk.Label(original_data_frame_inner, text=value).grid(row=i+1, column=j, padx=5, pady=2)
    original_data_frame_canvas.update_idletasks()
    original_data_frame_canvas.config(scrollregion=original_data_frame_canvas.bbox("all"))

# Function to update anonymized data frame
def update_anonymized_data_frame():
    # Anonymize the original dataset
    anonymized_data = anonymize_data(df.values[:100])  # Limiting to 100 rows for faster processing
    for i, column in enumerate(df.columns):
        tk.Label(anonymized_data_frame_inner, text=column).grid(row=0, column=i, padx=5, pady=2)
    for i, row in enumerate(anonymized_data):
        for j, value in enumerate(row):
            tk.Label(anonymized_data_frame_inner, text=value).grid(row=i+1, column=j, padx=5, pady=2)
    anonymized_data_frame_canvas.update_idletasks()
    anonymized_data_frame_canvas.config(scrollregion=anonymized_data_frame_canvas.bbox("all"))

# Create GUI window
window = tk.Tk()
window.title("Data Anonymization and Evaluation")

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(window)
notebook.pack(padx=10, pady=10, fill='both', expand=True)

# Create frames for each tab
original_data_frame = tk.Frame(notebook)
anonymized_data_frame = tk.Frame(notebook)
evaluation_frame = tk.Frame(notebook)

# Add tabs to the notebook
notebook.add(original_data_frame, text="Original Data")
notebook.add(anonymized_data_frame, text="Anonymized Data")
notebook.add(evaluation_frame, text="Evaluation Statistics")

# Add Canvas and Scrollbar to original data tab
original_data_frame_canvas = tk.Canvas(original_data_frame)
original_data_frame_canvas.pack(side=tk.LEFT, fill='both', expand=True)
original_data_frame_scrollbar = tk.Scrollbar(original_data_frame, orient=tk.VERTICAL, command=original_data_frame_canvas.yview)
original_data_frame_scrollbar.pack(side=tk.RIGHT, fill='y')
original_data_frame_canvas.configure(yscrollcommand=original_data_frame_scrollbar.set)
original_data_frame_inner = tk.Frame(original_data_frame_canvas)
original_data_frame_canvas.create_window((0, 0), window=original_data_frame_inner, anchor='nw')

# Add Canvas and Scrollbar to anonymized data tab
anonymized_data_frame_canvas = tk.Canvas(anonymized_data_frame)
anonymized_data_frame_canvas.pack(side=tk.LEFT, fill='both', expand=True)
anonymized_data_frame_scrollbar = tk.Scrollbar(anonymized_data_frame, orient=tk.VERTICAL, command=anonymized_data_frame_canvas.yview)
anonymized_data_frame_scrollbar.pack(side=tk.RIGHT, fill='y')
anonymized_data_frame_canvas.configure(yscrollcommand=anonymized_data_frame_scrollbar.set)
anonymized_data_frame_inner = tk.Frame(anonymized_data_frame_canvas)
anonymized_data_frame_canvas.create_window((0, 0), window=anonymized_data_frame_inner, anchor='nw')

# Update original and anonymized data frames
update_original_data_frame()
update_anonymized_data_frame()

# Add labels for evaluation statistics
pruning_power_label = tk.Label(evaluation_frame, text="Pruning Power: N/A")
pruning_power_label.grid(row=0, column=0, padx=5, pady=2)
precision_label = tk.Label(evaluation_frame, text="Precision: N/A")
precision_label.grid(row=1, column=0, padx=5, pady=2)
cpu_time_label = tk.Label(evaluation_frame, text="CPU Time: N/A")
cpu_time_label.grid(row=2, column=0, padx=5, pady=2)
communication_cost_label = tk.Label(evaluation_frame, text="Communication Cost: N/A")
communication_cost_label.grid(row=3, column=0, padx=5, pady=2)
gui_load_time_label = tk.Label(evaluation_frame, text=f"GUI Load Time: {time.time() - start_gui_time:.2f} seconds")
gui_load_time_label.grid(row=4, column=0, padx=5, pady=2)

# Run the GUI
window.mainloop()
