# Import all the libraries needed
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

# Loads dataset from a csv file (specificaly the columns "text and "label)
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data["text"], data["label"]

# Filter the data
def clean_text(text):
    if not isinstance(text, (str, bytes)): # if null returns empty string
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text) #using regex to remove special charachters
    text = text.lower().strip()
    return text


# Split dataset
def split_data(text, label):
    X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Loads and preprocesses the data
file_name = "train.csv" #change to file name
text, label = load_data(file_name)
text = text.apply(clean_text)
X_train, X_test, y_train, y_test = split_data(text, label)

