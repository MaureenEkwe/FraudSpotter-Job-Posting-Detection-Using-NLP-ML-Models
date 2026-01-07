# FraudSpotter: Job Posting Detection Using NLP & ML Models

## Overview
FraudSpotter detects fraudulent job listings by analyzing both the language used in postings and their structured metadata. This project compares traditional TF-IDF baseline models with a DistilBERT-based approach to determine whether deeper text understanding can improve scam detection.

## Dataset
- Source: Kaggle — Real vs Fake Job Posting Dataset  
  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- Contains 17,880 job postings with 18 columns, including job title, location, company profile, description, requirements, benefits, industry, and a binary `fraudulent` category.

## Libraries Used
This project was built using the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `xgboost`
- `imbalanced-learn` (for oversampling)
- `shap`
- `torch`
- `re`
- `os`
- `transformers`
- `tqdm`


## Installation
### Option 1: Using conda

conda install pandas numpy matplotlib scikit-learn scipy tqdm
conda install -c conda-forge xgboost imbalanced-learn shap transformers
conda install pytorch cpuonly -c pytorch

### Option 2: Using pip
pip install pandas numpy matplotlib scikit-learn scipy xgboost imbalanced-learn shap torch transformers tqdm

## Key Steps
- Load and clean the Kaggle job posting dataset
- Combine multiple job-related text fields into one feature
- Split data into training and testing sets
- Encode relevant structured job posting fields
- Balance classes using RandomOversampler
- Compare baseline models to a BERT-based approach
- SHAP analysis

## Files
### machinelearning_fraudspotter.py 
- TF-IDF representation of job text
- Combine encoded features + TF-IDF features
- Train and evaluate XGBoost 
- Scale features for Logistic Regression and KNN
- Interpret Logistic regression model features using SHAP


### BERT_fraudspotter.py (may take up to 2 hrs to run)
- Tokenize text to create features
- Generate contextual embeddings using DistilBERT
- Concatenate embeddings with structured features
- Scale and train Logistic Regression


##  Results
- **Top Models:** Logistic Regression & XGBoost: accuracy: ~ 0.987, ROC-AUC : ~ 0.99
- Strong and balanced fraud-class performance (F1: **~ 0.86**)

- Logistic Regression was selected as the primary baseline model because it offered competitive performance while remaining highly interpretable and better suited for SHAP's feature analysis.

## Files
- `machinelearning_fraudspotter.py` (TF-IDF models + SHAP analysis)
- `BERT_fraudspotter.py`  DistilBERT embeddings + Logistic Regression for performance metrics)


## Authors
Maureen Ekwebelem & YaeJin(Sally) Kang
