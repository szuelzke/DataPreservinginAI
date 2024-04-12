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

# Define your anonymization method (placeholder)
def anonymize_data(input_data):
  # Convert the NumPy array to a pandas DataFrame
  input_df = pd.DataFrame(input_data, columns=df.columns)

  # Deep copy the input data to avoid modifying the original DataFrame
  anonymized_data = input_df.copy()

  # Retrieve the index of the 'Name' and 'Age' columns
  name_index = input_df.columns.get_loc('Name')
  age_index = input_df.columns.get_loc('Age')
  discharge_date_index = input_df.columns.get_loc('Discharge Date')
  admission_date_index = input_df.columns.get_loc('Date of Admission')

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
  
  # Anonymize the 'Discharge Date' column (value is now just the year)
  anonymized_data.iloc[:, discharge_date_index] = anonymized_data.iloc[:, discharge_date_index].apply(lambda x: str(pd.to_datetime(x).year) if pd.notnull(x) else '')

  # Anonymize the 'Date of Admission' column (value is now just the year)
  anonymized_data.iloc[:, admission_date_index] = anonymized_data.iloc[:, admission_date_index].apply(lambda x: str(pd.to_datetime(x).year) if pd.notnull(x) else '')

  # Encode categorical variables: Doctor, Hospital, Insurance Provider
  categorical_columns = ['Doctor', 'Hospital', 'Insurance Provider']
  for column in categorical_columns:
        label_encoder = LabelEncoder()
        anonymized_data[column] = label_encoder.fit_transform(anonymized_data[column])

  
  # Convert 'Billing Amount' column to numeric and round to whole number
  anonymized_data['Billing Amount'] = pd.to_numeric(anonymized_data['Billing Amount'], errors='coerce').round()

  return anonymized_data

# Define evaluation and similarity functions
def evaluate_anonymized_data(original_data, anonymized_data, weights):
    overall_score, uniqueness_score, distribution_score, semantic_score = calculate_similarity(original_data, anonymized_data, weights)
    return overall_score, uniqueness_score, distribution_score, semantic_score

def calculate_similarity(original_data, anonymized_data, weights):
    # Distribution similarity
    distribution_similarity = calculate_distribution_similarity(original_data, anonymized_data)

    # Uniqueness preservation
    uniqueness_preservation = calculate_uniqueness_preservation(original_data, anonymized_data)

    # Semantic preservation
    semantic_preservation = calculate_semantic_preservation(original_data, anonymized_data)

    # Calculate overall similarity score using weighted sum
    overall_similarity = (weights['distribution'] * distribution_similarity +
                          weights['uniqueness'] * uniqueness_preservation +
                          weights['semantic'] * semantic_preservation)

    return overall_similarity, distribution_similarity, uniqueness_preservation, semantic_preservation

def calculate_distribution_similarity(original_data, anonymized_data):
    numerical_columns = original_data.select_dtypes(include=np.number).columns
    numerical_similarity_scores = []
    for column in numerical_columns:
        original_mean = original_data[column].mean()
        anonymized_mean = anonymized_data[column].mean()
        original_std = original_data[column].std()
        anonymized_std = anonymized_data[column].std()
        z_score_original = (original_data[column] - original_mean) / original_std
        z_score_anonymized = (anonymized_data[column] - anonymized_mean) / anonymized_std
        numerical_similarity_scores.append(1 - np.abs(z_score_original - z_score_anonymized).mean())
    categorical_columns = original_data.select_dtypes(exclude=np.number).columns
    categorical_similarity_scores = []
    for column in categorical_columns:
        original_value_counts = original_data[column].value_counts(normalize=True)
        anonymized_value_counts = anonymized_data[column].value_counts(normalize=True)
        common_categories = set(original_value_counts.index) & set(anonymized_value_counts.index)
        similarity_score = sum(min(original_value_counts[cat], anonymized_value_counts[cat]) for cat in common_categories)
        categorical_similarity_scores.append(similarity_score / min(original_value_counts.sum(), anonymized_value_counts.sum()))
    overall_similarity = (np.mean(numerical_similarity_scores) + np.mean(categorical_similarity_scores)) / 2
    return overall_similarity

def calculate_uniqueness_preservation(original_data, anonymized_data):
    uniqueness_scores = []
    for column in original_data.columns:
        original_unique_count = original_data[column].nunique()
        anonymized_unique_count = anonymized_data[column].nunique()
        uniqueness_scores.append(min(anonymized_unique_count / original_unique_count, 1.0))
    return np.mean(uniqueness_scores)

def calculate_semantic_preservation(original_data, anonymized_data):
    semantic_similarity_scores = []
    for column in original_data.columns:
        original_values = set(original_data[column])
        anonymized_values = set(anonymized_data[column])
        common_values = original_values.intersection(anonymized_values)
        similarity_score = len(common_values) / len(original_values)
        semantic_similarity_scores.append(similarity_score)
    return np.mean(semantic_similarity_scores)

# GUI setup
window = tk.Tk()
window.title("Data Anonymization and Evaluation")
notebook = ttk.Notebook(window)
notebook.pack(padx=10, pady=10, fill='both', expand=True)
original_data_frame = tk.Frame(notebook)
anonymized_data_frame = tk.Frame(notebook)
evaluation_frame = tk.Frame(notebook)
notebook.add(original_data_frame, text="Original Data")
notebook.add(anonymized_data_frame, text="Anonymized Data")
notebook.add(evaluation_frame, text="Evaluation Statistics")

# Set up original data frame
original_data_frame_canvas = tk.Canvas(original_data_frame)
original_data_frame_canvas.pack(side=tk.LEFT, fill='both', expand=True)
original_data_frame_scrollbar = tk.Scrollbar(original_data_frame, orient=tk.VERTICAL, command=original_data_frame_canvas.yview)
original_data_frame_scrollbar.pack(side=tk.RIGHT, fill='y')
original_data_frame_canvas.configure(yscrollcommand=original_data_frame_scrollbar.set)
original_data_frame_inner = tk.Frame(original_data_frame_canvas)
original_data_frame_canvas.create_window((0, 0), window=original_data_frame_inner, anchor='nw')

# Set up anonymized data frame
anonymized_data_frame_canvas = tk.Canvas(anonymized_data_frame)
anonymized_data_frame_canvas.pack(side=tk.LEFT, fill='both', expand=True)
anonymized_data_frame_scrollbar = tk.Scrollbar(anonymized_data_frame, orient=tk.VERTICAL, command=anonymized_data_frame_canvas.yview)
anonymized_data_frame_scrollbar.pack(side=tk.RIGHT, fill='y')
anonymized_data_frame_canvas.configure(yscrollcommand=anonymized_data_frame_scrollbar.set)
anonymized_data_frame_inner = tk.Frame(anonymized_data_frame_canvas)
anonymized_data_frame_canvas.create_window((0, 0), window=anonymized_data_frame_inner, anchor='nw')

# Update original and anonymized data frames
def update_original_data_frame():
    for i, column in enumerate(df.columns):
        tk.Label(original_data_frame_inner, text=column).grid(row=0, column=i, padx=5, pady=2)
    for i, row in enumerate(df.values[:100]):
        for j, value in enumerate(row):
            tk.Label(original_data_frame_inner, text=value).grid(row=i+1, column=j, padx=5, pady=2)
    original_data_frame_canvas.update_idletasks()
    original_data_frame_canvas.config(scrollregion=original_data_frame_canvas.bbox("all"))

def update_anonymized_data_frame():
    anonymized_data = anonymize_data(df.head(100))
    #anonymized_data = df.head(100)
    for i, column in enumerate(anonymized_data.columns):
        tk.Label(anonymized_data_frame_inner, text=column).grid(row=0, column=i, padx=5, pady=2)
    for i, row in anonymized_data.iterrows():
        for j, value in enumerate(row):
            tk.Label(anonymized_data_frame_inner, text=value).grid(row=i+1, column=j, padx=5, pady=2)
    anonymized_data_frame_canvas.update_idletasks()
    anonymized_data_frame_canvas.config(scrollregion=anonymized_data_frame_canvas.bbox("all"))

update_original_data_frame()
update_anonymized_data_frame()

# Update evaluation statistics
def update_evaluation_statistics():
    # Get weights from entry fields
    weights = {
        'uniqueness': float(uniqueness_weight_entry.get()),
        'distribution': float(distribution_weight_entry.get()),
        'semantic': float(semantic_weight_entry.get())
    }

    # Anonymize the original data
    anonymized_data = anonymize_data(df.head(100))
    #anonymized_data = df.head(100)

    # Evaluate anonymized data using the weights
    overall_score, uniqueness_score, distribution_score, semantic_score = evaluate_anonymized_data(df, anonymized_data, weights)

    # Update labels for evaluation statistics
    uniqueness_score_label.config(text="Uniqueness Score: {:.2f}".format(uniqueness_score))
    distribution_score_label.config(text="Distribution Score: {:.2f}".format(distribution_score))
    semantic_score_label.config(text="Semantic Score: {:.2f}".format(semantic_score))
    overall_score_label.config(text="Overall Score: {:.2f}".format(overall_score))

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

window.mainloop()
