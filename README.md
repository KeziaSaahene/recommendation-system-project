Recommendation System Project

This project focuses on designing and evaluating a personalized recommendation system using historical user interaction data. The CRISP-DM methodology guides the process from business understanding to deployment.

Tools & Technologies

Python – core programming language

Pandas, NumPy, SciPy – data manipulation & matrix factorization

Scikit-learn – preprocessing, evaluation metrics, anomaly detection

TensorFlow / Keras – deep learning (CNN & Autoencoder models)

Matplotlib & Seaborn – data visualization

Implicit (ALS) – collaborative filtering (if installed)

GitHub – version control & progress tracking

Dataset

The dataset contains three main files:

events.csv → user interactions (timestamp, visitorid, event, itemid, transactionid)

item_properties_combined.csv → item metadata (timestamp, itemid, categoryid, value)

category_tree.csv → hierarchical item relationships (child, parent)

Preprocessing Steps:

Converted timestamps (ms → datetime)

Cleaned encoded values (e.g., n123 → 123)

Normalized categorical IDs

Built user–item interaction matrix

📊 CRISP-DM Framework
1. Business Understanding

Objective:

Build a personalized recommendation system.

Detect abnormal users (bots, fraud, unusual patterns).



Business Questions:

1. How many unique visitors are interacting with the platform?
 Total number of visitors: 1407580

2. On average, how many events does a visitor generate?
  Average events per visitor: 1.96

3. How many unique items does a visitor typically interact with?
   Average unique items per visitor: 1.52
   
4. What are the peak hours when visitors are most active?
  7pm
   
5. What are the peak days of the week for visitor activity?
  Tuesday
   
6. How many parent categories exist on the platform?
   Total number of parent categories: 362
    
7. Which are the top parent categories with the most children?
  Top Parent Categories with Most Children: 250
    
8. How many categories exist in the dataset?
   Total number of categories: 1,092
    
9. How many total unique items are available across all categories?
  Total number of unique items: 5369k
 
10. On average, how many unique items are there per category?
  Average unique items per category: 4917.18

11. Which are the top 10 most interacted-with categories?


2. Data Understanding

Events dataset: 8.5M+ rows of user interactions

Item properties: metadata requiring latest-value extraction

Category tree: hierarchical grouping of items

Key Insights:

Events are skewed heavily toward views

Transactions form a small but valuable subset

Metadata preprocessing was crucial for labeling


3. Data Preparation
Loaded datasets (events.csv, category_tree.csv, item_properties_part1.csv, item_properties_part2.csv) using Dask for efficient handling of large files.

Combined both item_properties csvs(item_properties_part1.csv and item_properties_part2.csv) as one.

Explored dataset structure – checked shape, data types, missing values, duplicates, and unique counts across key identifiers (visitorid, itemid, transactionid).

Converted timestamps into datetime format and extracted additional features (date, hour, day of week, month, month name).

Summarized each dataset (`events.csv`, `category_tree.csv`, `item_properties.csv`) to understand structure, unique counts, and key patterns before analysis.



4. Modeling
CNN Model:
Modeling: Built a tuned CNN model on user-item interaction sequences using tokenized item properties.
Training: Applied early stopping and optimized with Adam.

Anomaly Detection:
Modeling: Built user-level behavioral features and trained a CNN Autoencoder for unsupervised anomaly detection.
Training: Used reconstruction loss (MSE) with early stopping.


5. Evaluation
For CNN Model

Evaluation: Assessed model with accuracy, precision, recall, F1-score, and ranking metrics (Recall@K, Hit Rate@K, NDCG@K).

For Anomaly Detection:

Evaluation: Flagged abnormal users based on high reconstruction error (top 2% threshold).

Error distributions visualized to highlight anomalies

6. Deployment / Deliverables
Recommender System: Saved trained CNN model, tokenizer, and label encoder as reusable artifacts.

Anomaly Detection: Saved CNN Autoencoder model and feature scaler for consistent scoring.

These artifacts enable future inference without retraining.


