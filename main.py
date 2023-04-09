# Imports all the libraries needed
import os
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import class_weight
import chardet

# Loads dataset from a csv file (specifically the columns "text" and "label")
def load_data(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())

    with open(file_path, "r", encoding=result["encoding"], errors='replace') as f:
        data = pd.read_csv(f)

    data["label"] = data["label"].astype(str)
    
    # Remove rows with null text or label
    data = data.dropna(subset=["text", "label"])
    
    # Filter only rows with labels '0' and '1'
    data = data[data["label"].isin(["0", "1"])]

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

def train_and_evaluate():
    # applies cross validation
    n_splits = 5
    kfold = get_stratified_kfold_splits(text, label, n_splits=n_splits)

    # Initializes scores
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    fold = 1
    for train_index, test_index in kfold:
        print(f"Training on fold {fold}/{n_splits}...")
        X_train, X_test = text.iloc[train_index], text.iloc[test_index]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]
        
        # Calculates class weights for current fold and updates the classifier with the new class weights
        class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights_dict = {str(c): w for c, w in zip(np.unique(y_train), class_weights)}
        clf = LogisticRegression(max_iter=5000, class_weight=class_weights_dict, random_state=42)


        #  extracts features
        X_train_features = extract_features(X_train, tokenizer, model, device)
        X_test_features = extract_features(X_test, tokenizer, model, device)
        
        # Trains the classifier
        clf.fit(X_train_features, y_train)

        # Evaluates model
        y_pred = clf.predict(X_test_features)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, pos_label='1'))
        recall_scores.append(recall_score(y_test, y_pred, pos_label='1'))
        f1_scores.append(f1_score(y_test, y_pred, pos_label='1'))


        print(f"Accuracy for fold {fold}: {accuracy_scores[-1]:.2%}")
        fold += 1


    # Evaluates model and prints stats
    print("Average Accuracy:", np.mean(accuracy_scores))
    print("Average Precision:", np.mean(precision_scores))
    print("Average Recall:", np.mean(recall_scores))
    print("Average F1 Score:", np.mean(f1_scores))


# Prediction pipeline meathod
def predict(text, tokenizer, model, device, clf):
    text = clean_text(text)
    features = extract_features([text], tokenizer, model, device)
    probabilities = clf.predict_proba(features)[0]
    
    # Get the class with the highest probability
    prediction = clf.predict(features)[0]
    prediction_probability = max(probabilities)
    
    return bool(prediction), prediction_probability


def test_model_on_custom_dataset(custom_dataset, tokenizer, model, device, clf):
    # Preprocess and clean the dataset
    custom_dataset["text"] = custom_dataset["text"].apply(clean_text)

    # Extract features from the custom dataset
    custom_dataset_features = extract_features(custom_dataset["text"], tokenizer, model, device)

    # Make predictions using the classifier
    custom_dataset_predictions = clf.predict(custom_dataset_features)

    return custom_dataset_predictions


# Initializes DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda being used: " + str(torch.cuda.is_available()))

model.to(device)

# Loads and preprocesses the data
file_name = "final_training_news_articles.csv"
text, label = load_data(file_name)

# Initialize the classifier
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(label.astype(str)), y=label.astype(str))
class_weights_dict = {str(c): w for c, w in zip(np.unique(label), class_weights)}
clf = LogisticRegression(max_iter=5000, class_weight=class_weights_dict, random_state=42)

modelName = 'model.pth'

if not os.path.isfile(modelName):
    train_and_evaluate()
    torch.save(model.state_dict(), modelName)
else:
    model.load_state_dict(torch.load(modelName))


file_name = "sample_final_testing.csv"
# Loads and preprocesses the custom dataset
# Loads and preprocesses the custom dataset
test_dataset = pd.read_csv(file_name)
test_dataset["label"] = test_dataset["label"].astype(str)


# Initialize the classifier
test_dataset_labels_str = test_dataset["label"].astype(str)
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(test_dataset_labels_str), y=test_dataset_labels_str)
class_weights_dict = {str(c): w for c, w in zip(np.unique(test_dataset["label"]), class_weights)}
clf = LogisticRegression(max_iter=5000, class_weight=class_weights_dict, random_state=42)

# Fit the classifier using the entire dataset
X_all = extract_features(text, tokenizer, model, device)
clf.fit(X_all, label)

# Test the model on the custom dataset
custom_dataset_predictions = test_model_on_custom_dataset(test_dataset, tokenizer, model, device, clf)
test_dataset_labels_str = test_dataset["label"].astype(str)
accuracy = accuracy_score(test_dataset_labels_str, custom_dataset_predictions)
precision = precision_score(test_dataset_labels_str, custom_dataset_predictions, pos_label='1')
recall = recall_score(test_dataset_labels_str, custom_dataset_predictions, pos_label='1')
f1 = f1_score(test_dataset_labels_str, custom_dataset_predictions, pos_label='1')

print("\nCustom dataset statistics:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")


while True:
    user_input = input("\nEnter your News prompt and it will tell you whether it is true or not, press 'stop' to exit: \n")
    if user_input == "stop":
        break
    prediction, prediction_probability = predict(user_input, tokenizer, model, device, clf)
    print("Prediction:", prediction)
    print("Prediction probability: {:.2%}".format(prediction_probability))
