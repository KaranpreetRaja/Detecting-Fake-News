# Imports all the libraries needed
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import DistilBertTokenizer, DistilBertModel

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

# Splits dataset
def split_data(text, label, test_size=0.2):
    data = pd.concat([text, label], axis=1)
    train_data = data.sample(frac=1 - test_size, random_state=42)
    test_data = data.drop(train_data.index)
    return train_data["text"], test_data["text"], train_data["label"], test_data["label"]

# Tokenization and feature extraction
def extract_features(texts, tokenizer, model, device):
    features = []
    for text in texts:
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
text = text.apply(clean_text)
X_train, X_test, y_train, y_test = split_data(text, label)

# Initializes DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Extracts features
X_train_features = extract_features(X_train, tokenizer, model, device)
X_test_features = extract_features(X_test, tokenizer, model, device)

# Trains classifier
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train_features, y_train)

# Evaluates model and prints stats
y_pred = clf.predict(X_test_features)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

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
