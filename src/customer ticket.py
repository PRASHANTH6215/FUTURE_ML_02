# SUPPORT TICKET CLASSIFICATION USING MACHINE LEARNING

# Import required libraries
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


# Download stopwords dataset for NLP preprocessing
nltk.download('stopwords')


# Load the dataset
print("Loading dataset...")

df = pd.read_csv(r"c:\Users\PRASHANTH\OneDrive\文档\Desktop\FUTURE_TASK_02\DATASET\customer_support_tickets.csv")

# Display first few rows of dataset
print(df.head())

# Display column names for reference
print("\nColumns in dataset:")
print(df.columns)


# Remove rows where important fields are missing
df = df.dropna(subset=["Ticket Description","Ticket Type","Ticket Priority"])


# Load English stopwords for text cleaning
stop_words = set(stopwords.words('english'))


# Function to clean ticket text
def clean_text(text):

    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Split text into words
    words = text.split()

    # Remove common stopwords
    words = [word for word in words if word not in stop_words]

    # Join cleaned words back into a sentence
    return " ".join(words)


print("\nCleaning text...")

# Apply text cleaning to ticket description
df["clean_text"] = df["Ticket Description"].apply(clean_text)


print("\nVectorizing text using TF-IDF...")

# Convert text data into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_text"])


print("\nTraining Category Model...")

# Target variable for ticket category classification
y_category = df["Ticket Type"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_category, test_size=0.2, random_state=42
)

# Train Logistic Regression model for category prediction
model_category = LogisticRegression(max_iter=1000)

model_category.fit(X_train, y_train)

# Predict categories on test data
pred_category = model_category.predict(X_test)

print("\nCATEGORY MODEL RESULTS")

# Display model accuracy and classification metrics
print("Accuracy:", accuracy_score(y_test, pred_category))

print(classification_report(y_test, pred_category))


print("\nTraining Priority Model...")

# Target variable for priority prediction
y_priority = df["Ticket Priority"]

# Split dataset for priority prediction
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

# Train Logistic Regression model for priority prediction
model_priority = LogisticRegression(max_iter=1000)

model_priority.fit(X_train2, y_train2)

# Predict priority on test data
pred_priority = model_priority.predict(X_test2)

print("\nPRIORITY MODEL RESULTS")

# Display model accuracy and evaluation metrics
print("Accuracy:", accuracy_score(y_test2, pred_priority))

print(classification_report(y_test2, pred_priority))


print("\nGenerating confusion matrix...")

# Confusion matrix to visualize classification performance
cm = confusion_matrix(y_test, pred_category)

plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Category Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# Function to predict category and priority for a new ticket
def predict_ticket(ticket):

    # Clean input ticket text
    cleaned = clean_text(ticket)

    # Convert text into TF-IDF vector
    vector = vectorizer.transform([cleaned])

    # Predict category and priority
    category = model_category.predict(vector)[0]

    priority = model_priority.predict(vector)[0]

    # Display prediction results
    print("\n==============================")

    print("Ticket:", ticket)

    print("Predicted Category:", category)

    print("Predicted Priority:", priority)

    print("==============================")


# Test the system with example tickets
predict_ticket("My internet is not working since yesterday")

predict_ticket("Payment failed but money was deducted")

predict_ticket("I cannot login to my account")