# DA5401 Data Challenge: Advanced Metric Learning for AI Evaluation Score Prediction

**Name:** Saranath P  
**Roll No:** DA25E003

## Executive Summary

This project presents a comprehensive approach to predicting fitness scores (0-10) for AI evaluation metric definitions and prompt-response pairs. The solution leverages advanced metric learning techniques, combining semantic embeddings, sophisticated feature engineering, and gradient boosting (LightGBM) to achieve strong predictive performance. Key strategies include strategic data augmentation to handle class imbalance and multilingual text processing.

## 1. Introduction and Problem Understanding

The goal is to predict a fitness score indicating how well a specific evaluation metric applies to a given user prompt and AI response.

**Key Inputs:**
*   **Metric Definitions:** Criteria for evaluating AI systems.
*   **Prompt-Response Pairs:** Conversational exchanges.
*   **System Prompts:** Context-setting instructions.

**Challenges Addressed:**
*   **Multilingual Complexity:** Handling mixed languages (English and Indian languages).
*   **Semantic Matching:** Capturing abstract relationships between metrics and conversations.
*   **Score Imbalance:** Addressing the skew towards high scores (8-10).
*   **Context Dependency:** Accounting for conversational context.

## 2. Methodology

### 2.1 Exploratory Data Analysis (EDA)
*   **Dataset:** 5,000 training samples, 3,638 test samples.
*   **Score Distribution:** Heavily skewed towards high scores (8-10), with very few low scores.
*   **Language Diversity:** Identified significant presence of non-English text.
*   **Text Analysis:** Analyzed length distributions of prompts and responses.

### 2.2 Feature Engineering
A rich feature set (3,073 dimensions) was constructed to capture multi-perspective relationships:
*   **Semantic Embeddings:** Generated using **LaBSE (Language-agnostic BERT Sentence Embedding)** to handle multilingual text effectively.
    *   Embeddings for Metric, Response, User Prompt, and System Prompt (768 dimensions each).
*   **Interaction Features:** Calculated Cosine Similarity between:
    *   Response and Metric
    *   User Prompt and Metric
    *   System Prompt and Metric

### 2.3 Data Augmentation
To address the severe class imbalance (scarcity of low scores), **Negative Sampling** was implemented:
*   Created synthetic low-scoring examples by pairing prompts/responses with irrelevant metrics.
*   Assigned low scores (0-3) to these mismatched pairs.
*   This strategy significantly improved the model's ability to distinguish between relevant and irrelevant metrics.

### 2.4 Modeling
*   **Model:** **LightGBM Regressor**
*   **Rationale:** Chosen for its efficiency and superior performance with high-dimensional, mixed-type feature spaces.
*   **Training:** Trained with RMSE loss, using early stopping to prevent overfitting.
*   **Hyperparameter Tuning:** Optimized parameters (learning rate, tree depth, regularization) using GridSearchCV.

## 3. Results

*   **Feature Importance:**
    *   Metric Embeddings: ~40% importance
    *   Response Embeddings: ~30% importance
    *   Similarity Features: ~20% importance
*   **Performance:** The model demonstrated robust performance on the validation set, effectively predicting fitness scores across the diverse test cases.

## 4. Conclusion

The project successfully demonstrates that combining traditional machine learning (LightGBM) with modern semantic embeddings (LaBSE) is a powerful approach for AI evaluation tasks. The strategic use of negative sampling was crucial in overcoming data imbalance, resulting in a robust and scalable solution.

## 5. How to Run

1.  **Prerequisites:** Python 3.x, Jupyter Notebook.
2.  **Dependencies:** Install required libraries (pandas, numpy, scikit-learn, sentence-transformers, lightgbm, matplotlib, seaborn, tqdm).
3.  **Data:** Ensure `train_data.json`, `test_data.json`, and `metric_names.json` are in the same directory.
4.  **Execution:** Run the `da25e003.ipynb` notebook to reproduce the analysis, training, and prediction generation.
