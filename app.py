import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# -------------------------
# Load artifacts
# -------------------------

# Load CNN model
model = tf.keras.models.load_model("cnn_model.h5")

# Load tokenizer
with open("tokenizer.json") as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

# Load label encoder
with open("labelencoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------
# Streamlit app
# -------------------------
st.title("📊 Text Classification App")
st.write("Enter text and get a prediction from the CNN model!")

# Input box for user
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip():
        # Convert text to sequences
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=100)  # adjust maxlen to your training value

        # Make prediction
        prediction = model.predict(padded)
        pred_label = np.argmax(prediction, axis=1)
        result = label_encoder.inverse_transform(pred_label)

        st.success(f"✅ Prediction: {result[0]}")
    else:
        st.warning("⚠️ Please enter some text!")
