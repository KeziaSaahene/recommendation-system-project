import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from collections import Counter


# Load dataset
def clean_value(val):
    """Convert 'n123' -> 123 (int), keep others unchanged."""
    if isinstance(val, str) and val.startswith("n"):
        try:
            return int(val[1:])
        except ValueError:
            return val
    return val

MAX_EVENTS = None  # limit for demo speed

# Load events
events = pd.read_csv(base_path + "events.csv", nrows=MAX_EVENTS)
# Load and clean item_properties
item_props = pd.read_csv(base_path + "item_properties_combined.csv",
                         names=["timestamp", "itemid", "property", "value"],
                         skiprows=1)  # Skip the header row
item_props["value"] = item_props["value"].apply(clean_value)

# Load category_tree
category_tree = pd.read_csv(base_path + "category_tree.csv",
                            names=["child", "parent"],
                            skiprows=1)  # also skip header row if needed

print(f"Events shape: {events.shape}")
print(f"Item properties shape: {item_props.shape}")
print(f"Category tree shape: {category_tree.shape}")

# Converting Events timestamp to datetime
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')

# Rename "property" to "categoryid"
item_props = item_props.rename(columns={"property": "categoryid"})

# Show column names
print("Events columns:", events.columns)
print("Items columns:", item_props.columns)
print("Category Tree columns:", category_tree.columns)

# Preview
print("Events sample:\n", events.head())
print(events.dtypes)
print("\nItems sample:\n", item_props.head())
print(item_props.dtypes)
print("\nCategory Tree sample:\n", category_tree.head())
print(category_tree.dtypes)

# Candidate Generation
# -------------------------------
def build_user_item_matrix(events_df):
    weight_map = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}
    events_df["weight"] = events_df["event"].map(weight_map).fillna(1.0)
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    events_df["user_idx"] = user_enc.fit_transform(events_df["visitorid"])
    events_df["item_idx"] = item_enc.fit_transform(events_df["itemid"])
    rows, cols, vals = events_df["user_idx"].values, events_df["item_idx"].values, events_df["weight"].values
    ui_matrix = sparse.coo_matrix((vals, (rows, cols)),
                                  shape=(len(user_enc.classes_), len(item_enc.classes_)))
    return ui_matrix.tocsr(), user_enc, item_enc

ui_matrix, user_enc, item_enc = build_user_item_matrix(events)

# ALS recommendations (if implicit is installed)
if implicit is not None:
    als_model = implicit.als.AlternatingLeastSquares(factors=32, iterations=5)
    als_model.fit(ui_matrix.T)
    user0_recs = als_model.recommend(0, ui_matrix[0], N=5)
    print("ALS recs for first user:", user0_recs)
else:
    print("implicit not installed, skipping ALS.")

# Evaluation Metrics
# -------------------------------
def recall_at_k(true_items, pred_items, k):
    return len(set(pred_items[:k]) & set(true_items)) / max(1, len(set(true_items)))

def hit_rate_at_k(true_items, pred_items, k):
    return int(len(set(pred_items[:k]) & set(true_items)) > 0)

def ndcg_at_k(true_items, pred_items, k):
    def dcg(scores):
        return sum((2**s - 1) / np.log2(i+2) for i, s in enumerate(scores))
    rels = [1 if p in true_items else 0 for p in pred_items[:k]]
    return dcg(rels) / max(dcg(sorted(rels, reverse=True)), 1e-9)

# Task 1 - Deep Learning Model (Tuned CNN Only + Evaluation Metrics)
# -------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import numpy as np
from collections import Counter

# --- Step 1: Prepare data ---
target_prop = item_props["categoryid"].value_counts().index[0]
print("Target property for prediction:", target_prop)

# Build item->latest value for target_prop
prop_df = item_props[item_props["categoryid"] == target_prop].sort_values("timestamp")
item_latest_prop = prop_df.groupby("itemid")["value"].last().to_dict()

X_texts, y_labels = [], []
for _, row in events[events["event"] == "addtocart"].iterrows():
    user_hist = events[(events["visitorid"] == row["visitorid"]) &
                       (events["timestamp"] < row["timestamp"]) &
                       (events["event"] == "view")]
    tokens = [str(item_latest_prop.get(i, "")) for i in user_hist["itemid"].values]
    label = item_latest_prop.get(row["itemid"])
    if label and tokens:
        X_texts.append(" ".join(tokens))
        y_labels.append(label)

# Filter to frequent classes only
counts = Counter(y_labels)
min_freq = 10
top_classes = {cls for cls, cnt in counts.items() if cnt >= min_freq}
filtered = [(txt, lbl) for txt, lbl in zip(X_texts, y_labels) if lbl in top_classes]

if not filtered:
    print("No training samples found after filtering.")
else:
    X_texts, y_labels = zip(*filtered)
    X_texts, y_labels = list(X_texts), list(y_labels)

    # Encode labels
    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y_labels)
    num_classes = len(label_enc.classes_)

    # Tokenize & pad sequences
    max_words = 5000
    max_len = 50
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_texts)
    X_seq = tokenizer.texts_to_sequences(X_texts)
    X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post')

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_pad, y_enc, test_size=0.2, random_state=42)

    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    # --- Step 2: Define Tuned CNN ---
    def build_cnn():
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(max_words, 128, input_length=max_len),
            tf.keras.layers.Conv1D(256, 5, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['accuracy']
        )        
        return model

    # --- Step 3: Train CNN ---
    cnn_model = build_cnn()
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    history = cnn_model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=8,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # --- Step 4: Evaluate ---
    loss, acc = cnn_model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"Tuned CNN Validation Accuracy: {acc:.4f}")

    # --- Step 5: Precision, Recall, F1 ---
    y_pred_probs = cnn_model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    precision = precision_score(y_val, y_pred, average="weighted")
    recall = recall_score(y_val, y_pred, average="weighted")
    f1 = f1_score(y_val, y_pred, average="weighted")

    print("\nClassification Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    #print("\nDetailed Report:\n", classification_report(y_val, y_pred, target_names=label_enc.classes_))

# --- Task 1: Save CNN Recommender Artifacts ---
import json
import pickle

# Save CNN model
cnn_model.save("cnn_model.h5")

# Save tokenizer
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as f:
    f.write(json.dumps(json.loads(tokenizer_json)))

# Save label encoder
with open("labelencoder.pkl", "wb") as f:
    pickle.dump(label_enc, f)

print("✅ Task 1 artifacts saved: cnn_model.h5, tokenizer.json, labelencoder.pkl")


# Anomally Detection 

# -------------------------------
# Task 2 - Abnormal User Detection (CNN Autoencoder + Visualization)
# -------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# --- Step 1: Build user-level features ---
def build_user_features(events_df):
    feats = []
    for uid, g in events_df.groupby("visitorid"):
        total = len(g)
        views = (g["event"] == "view").sum()
        adds = (g["event"] == "addtocart").sum()
        buys = (g["event"] == "transaction").sum()
        feats.append({
            "visitorid": uid,
            "total_events": total,
            "views": views,
            "adds": adds,
            "buys": buys,
            "add_rate": adds/total if total > 0 else 0,
            "conv_rate": buys/total if total > 0 else 0
        })
    return pd.DataFrame(feats).set_index("visitorid")

user_feats = build_user_features(events)

# --- Step 2: Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_feats)

# Reshape for CNN (samples, timesteps, channels)
X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# --- Step 3: Define CNN Autoencoder ---
def build_cnn_autoencoder(seq_len):
    input_layer = tf.keras.layers.Input(shape=(seq_len, 1))
    x = tf.keras.layers.Conv1D(32, 2, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
    x = tf.keras.layers.Conv1D(16, 2, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(32, 2, activation='relu', padding='same')(x)
    decoded = tf.keras.layers.Conv1D(1, 2, activation='linear', padding='same')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

cnn_autoencoder = build_cnn_autoencoder(X_cnn.shape[1])

# --- Step 4: Train autoencoder ---
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = cnn_autoencoder.fit(
    X_cnn, X_cnn,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# --- Step 5: Reconstruction error ---
reconstructions = cnn_autoencoder.predict(X_cnn)
mse = np.mean(np.power(X_cnn - reconstructions, 2), axis=(1, 2))
threshold = np.percentile(mse, 98)  # top 2% anomaly threshold

user_feats["reconstruction_error"] = mse
user_feats["outlier"] = (mse > threshold).astype(int)
print(f"Detected outliers: {user_feats['outlier'].sum()} users")

# --- Task 2: Save CNN Autoencoder Artifacts ---
import pickle

# Save CNN Autoencoder model
cnn_autoencoder.save("cnn_ae.h5")

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Task 2 artifacts saved: cnn_ae.h5, scaler.pkl")

# --- Step 6: Visualization (No Scientific Notation) ---
plt.figure(figsize=(10,5))
sns.histplot(mse, bins=50, kde=True, color='blue')

plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')

plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Number of Users")

# Disable scientific notation on both axes
plt.ticklabel_format(style='plain', axis='x')
plt.ticklabel_format(style='plain', axis='y')

plt.legend()
plt.show()

# Scatter plot for better view (No Scientific Notation)
plt.figure(figsize=(10,5))
plt.scatter(range(len(mse)), mse, c=(mse > threshold), cmap='coolwarm', s=10)

plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("User Reconstruction Error (Outlier Detection)")
plt.xlabel("User Index")
plt.ylabel("Reconstruction Error (MSE)")

# Disable scientific notation
plt.ticklabel_format(style='plain', axis='x')
plt.ticklabel_format(style='plain', axis='y')

plt.legend()
plt.show()
