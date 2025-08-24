#  Recommendation System Project

This project is focused on designing and evaluating a personalized recommendation system using historical user interaction data. It applies the CRISP-DM methodology to guide the entire process.
Tools & Technologies

Python – core programming language

Pandas, NumPy, Scipy – data manipulation & matrix factorization

Scikit-learn – preprocessing, evaluation metrics, anomaly detection

TensorFlow / Keras – deep learning (CNN & Autoencoder models)

Matplotlib & Seaborn – visualization of insights & anomalies

Implicit library (ALS) – collaborative filtering (if installed)

GitHub – version control & progress tracking

Dataset

The dataset contains three main files:

events.csv → User interactions (timestamp, visitorid, event, itemid, transactionid)

item_properties_combined.csv → Item properties (timestamp, itemid, categoryid, value)

category_tree.csv → Hierarchical relationships between item categories (child, parent)

Preprocessing steps included:

Timestamp conversion (ms → datetime)

Cleaning encoded values (e.g., n123 → 123)

Normalization of categorical IDs

Building User–Item Interaction Matrix

CRISP-DM Framework
 Business Understanding

Objective: Build a personalized recommendation system and detect abnormal users.

Analytical Questions:

What are the most common user interactions (view, add-to-cart, purchase)?

Which products or categories are most frequently purchased?

What is the conversion rate from views → add-to-cart → purchase?

Can we predict the next product category a user is likely to purchase?

How can we recommend relevant items based on past behavior?

Are there abnormal users (e.g., bots, fraud patterns)?

How do anomaly detection methods compare (Isolation Forest vs CNN Autoencoder)?

 Data Understanding

Events dataset → 8.5M+ rows of user interaction logs

Item properties → product metadata with category & value mapping

Category tree → hierarchical grouping of items

Key insights:

Events are heavily skewed towards views.

Transactions form a small but valuable subset.

Item properties required latest-value extraction for labeling.

Data Preparation

Converted timestamps

Encoded users & items using LabelEncoder

Built User–Item Matrix with weighted interactions:

View = 1

Add-to-cart = 3

Purchase = 5

Extracted user-level behavioral features (views, adds, buys, conversion rate)

Generated training samples for CNN recommender from historical sequences

 Modeling

Recommendation Models

Collaborative Filtering (ALS) → implicit feedback model

CNN Classifier → predicts item categories based on user history

Anomaly Detection

Isolation Forest (baseline traditional ML)

CNN Autoencoder (deep learning anomaly detection)

 Evaluation

Recommendation Metrics

Recall@K

Hit Rate@K

NDCG@K

CNN Classifier Results

Accuracy: ~0.85 (validation)

Weighted Precision/Recall/F1 reported

Anomaly Detection Results

Outlier users detected using reconstruction error threshold (98th percentile)

Visualized distribution of errors and flagged anomalies

 Deployment / Deliverables

✅ cnn_model.h5 – Trained CNN recommender model

✅ cnn_ae.h5 – Trained CNN Autoencoder for anomaly detection

✅ Visualization plots:

Error distribution of anomalies

User–item recommendation insights
