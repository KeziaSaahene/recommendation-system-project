Recommendation System Project

This project focuses on designing and evaluating a personalized recommendation system using historical user interaction data. The CRISP-DM methodology guides the process from business understanding to deployment.

🔧 Tools & Technologies

Python – core programming language

Pandas, NumPy, SciPy – data manipulation & matrix factorization

Scikit-learn – preprocessing, evaluation metrics, anomaly detection

TensorFlow / Keras – deep learning (CNN & Autoencoder models)

Matplotlib & Seaborn – data visualization

Implicit (ALS) – collaborative filtering (if installed)

GitHub – version control & progress tracking

📂 Dataset

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

Analytical Questions:

What are the most common user interactions (view, add-to-cart, purchase)?

Which products/categories are most frequently purchased?

What is the conversion rate from views → add-to-cart → purchase?

Can we predict the next product category a user is likely to purchase?

How can we recommend relevant items based on past behavior?

How do anomaly detection methods compare (Isolation Forest vs CNN Autoencoder)?

2. Data Understanding

Events dataset: 8.5M+ rows of user interactions

Item properties: metadata requiring latest-value extraction

Category tree: hierarchical grouping of items

Key Insights:

Events are skewed heavily toward views

Transactions form a small but valuable subset

Metadata preprocessing was crucial for labeling

3. Data Preparation

Converted timestamps → datetime format

Encoded users & items using LabelEncoder

Built user–item matrix with weighted interactions:

View = 1

Add-to-cart = 3

Purchase = 5

Extracted user-level behavioral features (views, adds, buys, conversion rate)

Generated CNN training samples from user histories

4. Modeling

Recommendation Models:

Collaborative Filtering (ALS): implicit feedback model

CNN Classifier: predicts item categories from user history

Anomaly Detection:

Isolation Forest – traditional ML baseline

CNN Autoencoder – deep learning anomaly detection

5. Evaluation

Recommendation Metrics:

Recall@K

Hit Rate@K

NDCG@K

CNN Classifier Results:

Accuracy: ~0.85 (validation)

Reported weighted Precision, Recall, F1

Anomaly Detection Results:

Outliers flagged using reconstruction error threshold (98th percentile)

Error distributions visualized to highlight anomalies

6. Deployment / Deliverables

✅ cnn_model.h5 – trained CNN recommender model
✅ cnn_ae.h5 – trained CNN Autoencoder for anomaly detection
✅ Visualization plots:

Anomaly error distribution

User–item recommendation insights
