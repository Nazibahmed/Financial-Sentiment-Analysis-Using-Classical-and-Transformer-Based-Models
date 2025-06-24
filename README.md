# Financial Sentiment Analysis Using Classical and Transformer Models

## Project Overview

This project focuses on performing sentiment analysis of **financial news headlines** using various **machine learning models** and **transformer-based models**. The primary goal is to evaluate the effectiveness of both traditional models (Logistic Regression, SVM, and Random Forest) and deep learning models (**BERT** and **FinBERT**) for sentiment classification.

We aim to compare the performance of these models on a publicly available **financial news dataset** sourced from **Kaggle**, which includes headlines labeled with sentiment categories (positive, neutral, and negative).

## Dataset

The dataset used in this project is sourced from **Kaggle's Financial News Dataset**, which contains thousands of financial news headlines. The headlines are pre-labeled with sentiment categories, making it ideal for supervised learning tasks like sentiment classification.

### Sentiment Labels:
- **Positive** → 2
- **Neutral** → 1
- **Negative** → 0

### Features:
- **News Headline Text** (input feature for sentiment classification)

## Models Used

### 1. **Logistic Regression**
   - A classical machine learning algorithm used for sentiment classification.
   - Text is converted into numerical features using **TF-IDF**.

### 2. **SVM (Support Vector Machine)**
   - A supervised machine learning model used for binary and multi-class classification tasks.
   - Text is converted into numerical features using **TF-IDF**.

### 3. **Random Forest**
   - An ensemble learning method that combines multiple decision trees to improve classification accuracy.
   - Text is converted into numerical features using **TF-IDF**.

### 4. **BERT (Fine-tuned)**
   - A pre-trained transformer model fine-tuned on the **financial news dataset**.
   - Direct sentiment inference is performed from raw text, with the model adapted to understand the context of financial headlines.

### 5. **FinBERT**
   - A variant of **BERT** fine-tuned specifically for financial text.
   - Direct sentiment inference is performed from raw text, optimized for financial data.

## Project Structure

- `data/`: Contains the raw dataset (financial news headlines).
- `notebooks/`: Jupyter notebooks for data preprocessing, model training, and evaluation.
- `models/`: Python scripts for implementing and training each model.
- `results/`: Evaluation metrics, comparison tables, and graphs.
- `README.md`: Project overview and instructions.

## Evaluation Metrics

The performance of each model is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

## Installation and Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/financial-sentiment-analysis.git
    cd financial-sentiment-analysis
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the `data/` directory.

4. Run the Jupyter notebooks in the `notebooks/` directory to preprocess the data, train the models, and evaluate their performance.

## Model Training and Evaluation

1. Preprocess the dataset (convert text to lowercase, remove punctuation, URLs, and expand financial terms).
2. Train each model using the preprocessed data:
   - **Logistic Regression**, **SVM**, and **Random Forest** use **TF-IDF** for feature extraction.
   - **BERT (Fine-tuned)** and **FinBERT** perform direct inference from raw text.
3. Compare the models based on **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.

## Results

The performance of each model is compared to assess the effectiveness of **transformer models** (like **FinBERT**) vs **classical machine learning models** (like **Logistic Regression**, **SVM**, and **Random Forest**). Results are displayed in **tables** and **graphs**.

## Future Work

- Further tuning of **BERT** and **FinBERT** models on domain-specific financial data.
- Investigating ensemble learning approaches to combine the strengths of classical models and transformers.

