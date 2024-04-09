import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import LabelEncoder
import time

def load_data(file_path):
    return pd.read_csv(file_path)

def display_original_data(original_data_frame, original_frame):
    original_columns = original_data_frame.columns
    for column in original_columns:
        tk.Label(original_frame, text=column, padx=5, pady=2, relief="ridge", width=15).pack(side="left")
    tk.Frame(original_frame, height=1, bg="gray").pack(fill="x", pady=2)
    for index, row in original_data_frame.iterrows():
        for value in row:
            tk.Label(original_frame, text=value, padx=5, pady=2, relief="ridge", width=15).pack(side="left")
        tk.Frame(original_frame, height=1, bg="gray").pack(fill="x", pady=2)

def display_anonymized_data(anonymized_data_frame, anonymized_frame):
    anonymized_columns = anonymized_data_frame.columns
    for column in anonymized_columns:
        tk.Label(anonymized_frame, text=column, padx=5, pady=2, relief="ridge", width=15).pack(side="left")
    tk.Frame(anonymized_frame, height=1, bg="gray").pack(fill="x", pady=2)
    for index, row in anonymized_data_frame.iterrows():
        for value in row:
            tk.Label(anonymized_frame, text=value, padx=5, pady=2, relief="ridge", width=15).pack(side="left")
        tk.Frame(anonymized_frame, height=1, bg="gray").pack(fill="x", pady=2)

def main():
    # Load data
    original_data = load_data('healthcare_dataset.csv')
    anonymized_data = original_data.copy()
    
    # Create GUI
    root = tk.Tk()
    root.title("Healthcare Data Anonymization")
    
    # Create tabs
    tab_control = tk.ttk.Notebook(root)
    tab_control.pack(expand=1, fill="both")
    
    original_frame = tk.Frame(tab_control)
    anonymized_frame = tk.Frame(tab_control)
    evaluation_frame = tk.Frame(tab_control)
    
    tab_control.add(original_frame, text="Original Data")
    tab_control.add(anonymized_frame, text="Anonymized Data")
    tab_control.add(evaluation_frame, text="Evaluation Statistics")
    
    # Display original data
    original_canvas = tk.Canvas(original_frame)
    original_scrollbar = tk.Scrollbar(original_frame, orient="vertical", command=original_canvas.yview)
    original_scrollable_frame = tk.Frame(original_canvas)
    
    original_scrollable_frame.bind("<Configure>", lambda e: original_canvas.configure(scrollregion=original_canvas.bbox("all")))
    
    original_canvas.create_window((0, 0), window=original_scrollable_frame, anchor="nw")
    original_canvas.configure(yscrollcommand=original_scrollbar.set)
    
    display_original_data(original_data, original_scrollable_frame)
    
    original_canvas.pack(side="left", fill="both", expand=True)
    original_scrollbar.pack(side="right", fill="y")
    
    # Display anonymized data
    anonymized_canvas = tk.Canvas(anonymized_frame)
    anonymized_scrollbar = tk.Scrollbar(anonymized_frame, orient="vertical", command=anonymized_canvas.yview)
    anonymized_scrollable_frame = tk.Frame(anonymized_canvas)
    
    anonymized_scrollable_frame.bind("<Configure>", lambda e: anonymized_canvas.configure(scrollregion=anonymized_canvas.bbox("all")))
    
    anonymized_canvas.create_window((0, 0), window=anonymized_scrollable_frame, anchor="nw")
    anonymized_canvas.configure(yscrollcommand=anonymized_scrollbar.set)
    
    display_anonymized_data(anonymized_data, anonymized_scrollable_frame)
    
    anonymized_canvas.pack(side="left", fill="both", expand=True)
    anonymized_scrollbar.pack(side="right", fill="y")
    
    root.mainloop()

if __name__ == "__main__":
    main()
