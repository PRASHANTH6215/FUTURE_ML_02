# Support Ticket Classification using Machine Learning

## Project Overview

Customer support teams receive hundreds of support tickets daily.  
Manually categorizing and prioritizing tickets can be slow and inefficient.

This project builds a Machine Learning system that automatically:

• Classifies support tickets into categories  
• Predicts ticket priority levels  

This helps companies respond faster and improve customer satisfaction.

---

## Objectives

The goal of this project is to build an NLP-based ML system that:

1. Reads customer support ticket text
2. Classifies tickets into categories
3. Predicts priority levels (Low / Medium / High / Critical)

---

## Technologies Used

Python  
Scikit-learn  
NLTK (Natural Language Processing)  
TF-IDF Vectorization  
Logistic Regression  
Seaborn / Matplotlib for visualization

---

## Machine Learning Workflow

1. Load the dataset
2. Clean and preprocess ticket text
3. Convert text into numerical features using TF-IDF
4. Train machine learning classification models
5. Evaluate model performance using accuracy, precision, recall
6. Predict category and priority for new support tickets

---

## Text Preprocessing

The following preprocessing steps were applied:

• Convert text to lowercase  
• Remove punctuation  
• Remove stopwords using NLTK  
• Tokenize words  

Example:

Original Ticket:

"My internet connection has not been working since yesterday."

Cleaned Text:

"internet connection working since yesterday"

---

## Feature Extraction

Text data is converted into numerical vectors using **TF-IDF (Term Frequency – Inverse Document Frequency)**.

TF-IDF helps the model identify important words in a ticket.

---

## Machine Learning Models

Two classification models are trained:

### 1 Ticket Category Model

Predicts the type of issue:

• Billing  
• Technical Issue  
• Account Problem  
• General Inquiry  

Algorithm used:

Logistic Regression

---

### 2 Priority Prediction Model

Predicts ticket urgency:

• Low  
• Medium  
• High  
• Critical  

Algorithm used:

Logistic Regression

---

## Model Evaluation

The models are evaluated using:

• Accuracy  
• Precision  
• Recall  
• F1-score  
• Confusion Matrix  

Example output:

Category Classification Accuracy: 85%

Priority Prediction Accuracy: 80%

---

## Confusion Matrix

The confusion matrix helps visualize prediction errors between ticket categories.

---

## Example Prediction

Input Ticket:

"My payment failed but money was deducted."

Model Output:

Category: Billing  
Priority: High

---

## Business Impact

This system helps support teams:

• Automatically categorize tickets  
• Identify urgent issues faster  
• Reduce manual ticket sorting  
• Improve response time  

This approach is commonly used in SaaS platforms and IT service management systems.

---
## Example Prediction

Input Ticket:
"My internet connection is not working."

Output:
Category: Technical Issue  
Priority: High

## Example Prediction

Input Ticket:
"My internet connection is not working."

Output:
Category: Technical Issue  
Priority: High

## How to Run the Project

Install dependencies:
pandas
numpy
scikit-learn
nltk
seaborn
matplotlib

run the program:

---

## Future Improvements

• Use advanced NLP models like BERT  
• Deploy the model as a web API  
• Build a dashboard for ticket monitoring  
• Implement real-time ticket prediction

---

## Author

Prashanth B
