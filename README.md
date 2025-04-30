# Sentiment Analysis

## Overview

In this project, sentiment analysis is performed on product reviews. The goal is to classify the reviews into two categories: positive (rating > 3) and negative (rating <= 3). The project utilizes three different techniques for vectorizing the text data: TF-IDF, Word2Vec, and BERT. Additionally, hyperparameter tuning is performed for three classification models (K-Nearest Neighbors, Random Forest, and Logistic Regression) to maximize the F1 score for accurate predictions.

## Dataset

The dataset is provided on the course website. It consists of product reviews with the following key columns:
- `review`: The text content of the review.
- `rating`: The rating given by the user (numeric scale).
  
### Dataset Preprocessing
1. Convert the review text to lowercase.
2. Remove links from the text.
3. Remove stopwords and punctuation.
4. Ensure the text is in English.

The `rating` column is then transformed into a binary classification where ratings greater than 3 are marked as positive (1), and ratings 3 or below are marked as negative (0).

## Vectorization Methods

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
   - Converts the reviews into numerical form based on the frequency of terms while penalizing common words.

2. **Word2Vec**:
   - Word2Vec uses a neural network model to map words into high-dimensional vectors based on their context.

3. **BERT (Bidirectional Encoder Representations from Transformers)**:
   - A transformer-based model that generates contextualized word embeddings for more accurate text representations.

## Hyperparameter Tuning

For each vectorization method, hyperparameter tuning is performed using grid search for three classification models:
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Logistic Regression**

The goal of tuning is to maximize the F1 score, ensuring that the models perform effectively on unseen data.

## Models & Results

The project applies three models:
1. **K-Nearest Neighbors (KNN)**: A simple yet effective method that classifies a point based on the majority class of its neighbors.
2. **Random Forest Classifier**: An ensemble learning method that uses multiple decision trees to classify data.
3. **Logistic Regression**: A statistical model that is used for binary classification tasks.

After training and testing the models on the vectorized data, the best-performing model is selected based on the F1 score.

## Requirements

Before running the code, install the necessary Python packages:

```bash
pip install pandas numpy scikit-learn gensim sentence-transformers langdetect nltk
```
## How to Run the Code
1. Load and Preprocess Data: The dataset is loaded and preprocessed by cleaning the reviews (removing stopwords, punctuation, and links).

2. Vectorization: Reviews are transformed using TF-IDF, Word2Vec, and BERT.

3. Hyperparameter Tuning: Grid search is performed for each model to find the optimal hyperparameters.

4. Model Evaluation: The models are evaluated using the F1 score, and the results are displayed.

## Conclusion
This project demonstrates the process of performing sentiment analysis using three different text vectorization techniques. By optimizing hyperparameters, the models achieved an F1 score that exceeds 0.8, indicating effective performance in classifying the sentiment of the reviews.
