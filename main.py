# Imports all the libraries needed
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import class_weight


# Loads dataset from a csv file (specifically the columns "text" and "label")
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data["text"], data["label"]

# Filters the data
def clean_text(text):
    if not isinstance(text, (str, bytes)): # if null returns empty string
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text) #using regex to remove special characters
    text = text.lower().strip()
    return text

# applies k-fold cross-validation
def get_stratified_kfold_splits(text, label, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return skf.split(text, label)

# Tokenization and feature extraction
def extract_features(texts, tokenizer, model, device):
    features = []
    for text in texts:
        if not isinstance(text, str):
            text = ""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        features.append(embeddings[0])
    return np.vstack(features)


# Loads and preprocesses the data
file_name = "train.csv"
text, label = load_data(file_name)

# applies cross validation
n_splits = 5
kfold = get_stratified_kfold_splits(text, label, n_splits=n_splits)

# Initializes DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda being used: "+ torch.cuda.is_available())
model.to(device)


# Initializes scores
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kfold:
    X_train, X_test = text.iloc[train_index], text.iloc[test_index]
    y_train, y_test = label.iloc[train_index], label.iloc[test_index]
    
    # Calculates class weights for current fold and updates the classifier with the new class weights
    class_weights = class_weight.compute_sample_weight("balanced", y_train)
    class_weights = dict(enumerate(class_weights))
    clf = LogisticRegression(max_iter=100, class_weight=class_weights, random_state=42)

    #  extracts features
    X_train_features = extract_features(X_train, tokenizer, model, device)
    X_test_features = extract_features(X_test, tokenizer, model, device)
    
    # Trains the classifier
    clf.fit(X_train_features, y_train)

    # Evaluates model
    y_pred = clf.predict(X_test_features)
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, pos_label=1))
    recall_scores.append(recall_score(y_test, y_pred, pos_label=1))
    f1_scores.append(f1_score(y_test, y_pred, pos_label=1))


# Evaluates model and prints stats
print("Average Accuracy:", np.mean(accuracy_scores))
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))
print("Average F1 Score:", np.mean(f1_scores))


# Prediction pipeline meathod
def predict(text, tokenizer, model, device, clf):
    text = clean_text(text)
    features = extract_features([text], tokenizer, model, device)
    prediction = clf.predict(features)[0]
    return bool(prediction)

while True:
    user_input = input("\nEnter your News prompt and it will tell you weather it is true or not, press 'stop' to exit: \n")
    if user_input == "stop":
        break
    print("Prediction:", predict(user_input, tokenizer, model, device, clf))
