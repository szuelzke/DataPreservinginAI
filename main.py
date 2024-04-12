
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
  # Convert the NumPy array to a pandas DataFrame
  input_df = pd.DataFrame(input_data, columns=df.columns)

  # Deep copy the input data to avoid modifying the original DataFrame
  anonymized_data = input_df.copy()

  # Retrieve the index of the 'Name' and 'Age' columns
  name_index = input_df.columns.get_loc('Name')
  age_index = input_df.columns.get_loc('Age')

  # Placeholder values for anonymization
  placeholder_values = [f"Person_{i+1}" for i in range(len(anonymized_data))]

  # Convert the DataFrame to object type to avoid dtype issues
  anonymized_data = anonymized_data.astype(object)

  # Assign the placeholder values to the 'Name' column
  anonymized_data.iloc[:, name_index] = pd.Series(placeholder_values, dtype=str)

  # Anonymize the 'Age' column
  age_ranges = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
  for i, (start, end) in enumerate(age_ranges, start=1):
      anonymized_data.loc[(anonymized_data['Age'] >= start) & (anonymized_data['Age'] <= end), 'Age'] = i

  return anonymized_data


def evaluate_anonymized_data(original_data, anonymized_data, weights):
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
  anonymized_data = anonymize_data(df.head(100))  # Limiting to 100 rows for faster processing
  for i, column in enumerate(anonymized_data.columns):
      tk.Label(anonymized_data_frame_inner, text=column).grid(row=0, column=i, padx=5, pady=2)
  for i, row in anonymized_data.iterrows():
      for j, value in enumerate(row):
          tk.Label(anonymized_data_frame_inner, text=value).grid(row=i+1, column=j, padx=5, pady=2)
  anonymized_data_frame_canvas.update_idletasks()
  anonymized_data_frame_canvas.config(scrollregion=anonymized_data_frame_canvas.bbox("all"))

# Function to update evaluation statistics
def update_evaluation_statistics():
    # Get weights from entry fields
    weights = {
        'uniqueness': float(uniqueness_weight_entry.get()),
        'distribution': float(distribution_weight_entry.get()),
        'semantic': float(semantic_weight_entry.get())
    }

    # Evaluate anonymized data using the weights
    overall_score, uniqueness_score, distribution_score, semantic_score = evaluate_anonymized_data(df, df.copy(), weights)

    # Update labels for evaluation statistics
    uniqueness_score_label.config(text="Uniqueness Score: {:.2f}".format(uniqueness_score))
    distribution_score_label.config(text="Distribution Score: {:.2f}".format(distribution_score))
    semantic_score_label.config(text="Semantic Score: {:.2f}".format(semantic_score))
    overall_score_label.config(text="Overall Score: {:.2f}".format(overall_score))

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

# Add labels and entry fields for evaluation statistics
uniqueness_weight_label = tk.Label(evaluation_frame, text="Uniqueness Weight:")
uniqueness_weight_label.grid(row=0, column=0, padx=5, pady=2)
uniqueness_weight_entry = tk.Entry(evaluation_frame)
uniqueness_weight_entry.grid(row=0, column=1, padx=5, pady=2)
uniqueness_weight_entry.insert(tk.END, "0.3")

distribution_weight_label = tk.Label(evaluation_frame, text="Distribution Weight:")
distribution_weight_label.grid(row=1, column=0, padx=5, pady=2)
distribution_weight_entry = tk.Entry(evaluation_frame)
distribution_weight_entry.grid(row=1, column=1, padx=5, pady=2)
distribution_weight_entry.insert(tk.END, "0.5")

semantic_weight_label = tk.Label(evaluation_frame, text="Semantic Weight:")
semantic_weight_label.grid(row=2, column=0, padx=5, pady=2)
semantic_weight_entry = tk.Entry(evaluation_frame)
semantic_weight_entry.grid(row=2, column=1, padx=5, pady=2)
semantic_weight_entry.insert(tk.END, "0.2")

update_button = tk.Button(evaluation_frame, text="Update Weights", command=update_evaluation_statistics)
update_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

uniqueness_score_label = tk.Label(evaluation_frame, text="Uniqueness Score: ")
uniqueness_score_label.grid(row=4, column=0, padx=5, pady=2)
distribution_score_label = tk.Label(evaluation_frame, text="Distribution Score: ")
distribution_score_label.grid(row=5, column=0, padx=5, pady=2)
semantic_score_label = tk.Label(evaluation_frame, text="Semantic Score: ")
semantic_score_label.grid(row=6, column=0, padx=5, pady=2)
overall_score_label = tk.Label(evaluation_frame, text="Overall Score: ")
overall_score_label.grid(row=7, column=0, padx=5, pady=2)
gui_load_time_label = tk.Label(evaluation_frame, text=f"GUI Load Time: {time.time() - start_gui_time:.2f} seconds")
gui_load_time_label.grid(row=8, column=0, padx=5, pady=2)

# Run the GUI
window.mainloop()
