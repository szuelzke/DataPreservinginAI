import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import LabelEncoder
import time


# ----

def calculate_score(original_data, anonymized_data, weights):
    # Calculate uniqueness score
    uniqueness_score = calculate_uniqueness_score(original_data, anonymized_data)
    
    # Calculate distribution score
    distribution_score = calculate_distribution_score(original_data, anonymized_data)
    
    # Calculate semantic preservation score
    semantic_score = calculate_semantic_score(original_data, anonymized_data)
    
    # Combine scores using weights
    overall_score = (weights['uniqueness'] * uniqueness_score +
                     weights['distribution'] * distribution_score +
                     weights['semantic'] * semantic_score)
    
    return overall_score, uniqueness_score, distribution_score, semantic_score

# Uniqueness Score: Measures how many unique values are present in the anonymized dataset compared to the original dataset. 
# Higher uniqueness score indicates more unique values, which can enhance privacy.
def calculate_uniqueness_score(original_data, anonymized_data):
    unique_values_ratios = []
    for column in original_data.columns:
        unique_values_ratio = len(anonymized_data[column].unique()) / len(original_data[column].unique())
        unique_values_ratios.append(unique_values_ratio)
    
    # Average uniqueness score across columns
    uniqueness_score = sum(unique_values_ratios) / len(unique_values_ratios)
    return uniqueness_score

# Distribution Score: Assesses how closely the statistical distribution of data in the anonymized dataset matches that of the original dataset. 
# A lower distribution score suggests greater deviation from the original distribution, which may indicate potential information loss.
def calculate_distribution_score(original_data, anonymized_data):
    numeric_columns = original_data.select_dtypes(include=[int, float]).columns
    distribution_differences = []
    for column in numeric_columns:
        distribution_difference = abs(anonymized_data[column].mean() - original_data[column].mean()) + \
                                  abs(anonymized_data[column].median() - original_data[column].median()) + \
                                  abs(anonymized_data[column].std() - original_data[column].std())
        distribution_differences.append(distribution_difference)
    
    # Average distribution difference across numeric columns
    distribution_score = sum(distribution_differences) / len(distribution_differences)
    return distribution_score

# Semantic Score: Evaluates how well the meaning or content of data is preserved between the original and anonymized datasets. 
# A higher semantic score indicates better preservation of the original data's semantic information.
def calculate_semantic_score(original_data, anonymized_data):
    total_score = 0
    total_values = 0
    
    for column in original_data.columns:
        original_values = original_data[column]
        anonymized_values = anonymized_data[column]
        total_values += len(original_values)
        total_score += sum(original_values == anonymized_values)
    
    semantic_score = total_score / total_values if total_values != 0 else 0
    return semantic_score

 
# ----

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

    weights = {'uniqueness': 0.3, 'distribution': 0.5, 'semantic': 0.2}
    overall_score, uniqueness_score, distribution_score, semantic_score = calculate_score(original_data, anonymized_data, weights)

    return overall_score, uniqueness_score, distribution_score, semantic_score

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

# Evaluate anonymized data
overall_score, uniqueness_score, distribution_score, semantic_score = evaluate_anonymized_data(df, df.copy())

# Add labels for evaluation statistics

uniqueness_score_label = tk.Label(evaluation_frame, text="Uniqueness Score: {:.2f}".format(uniqueness_score))
uniqueness_score_label.grid(row=0, column=0, padx=5, pady=2)
distribution_score_label = tk.Label(evaluation_frame, text="Distribution Score: {:.2f}".format(distribution_score))
distribution_score_label.grid(row=1, column=0, padx=5, pady=2)
semantic_score_label = tk.Label(evaluation_frame, text="Semantic Score: {:.2f}".format(semantic_score))
semantic_score_label.grid(row=2, column=0, padx=5, pady=2)
overall_score_label = tk.Label(evaluation_frame, text="Overall Score: {:.2f}".format(overall_score))
overall_score_label.grid(row=3, column=0, padx=5, pady=2)
gui_load_time_label = tk.Label(evaluation_frame, text=f"GUI Load Time: {time.time() - start_gui_time:.2f} seconds")
gui_load_time_label.grid(row=4, column=0, padx=5, pady=2)

# Run the GUI
window.mainloop()
