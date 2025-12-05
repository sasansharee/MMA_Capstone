#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np

# Load the CatBoost model, scaler, and label encoder from your specified paths
model = joblib.load(r"spotify_best_cat_boosting_model.pkl")
scaler = joblib.load(r"scaler.pkl")
le = joblib.load(r"label_encoder.pkl")

# Manually specified feature ranges
feature_ranges = {
    "danceability": (0.0651, 0.988),
    "energy": (0.0223, 0.999),
    "key": (0, 11),
    "loudness": (-22.534, 1.313),
    "mode": (0, 1),
    "speechiness": (0.0232, 0.905),
    "acousticness": (0.000003, 0.976),
    "instrumentalness": (0.0, 0.981),
    "liveness": (0.0205, 0.988),
    "valence": (0.0218, 0.976),
    "tempo": (64.934, 220.138),
    "duration_ms": (43807, 752000),
    "time_signature": (1, 5)
}

# Create a mapping from class indices to genre names
genre_map = {idx: label for idx, label in enumerate(le.classes_)}

st.title("Spotify Genre Prediction")

# Show feature ranges in the sidebar
st.sidebar.header("Feature Ranges (from dataset)")
for feature, (min_val, max_val) in feature_ranges.items():
    st.sidebar.write(f"{feature.capitalize()}: {min_val:.3f} to {max_val:.3f}")

# Input widgets for audio features using dynamic min/max values and appropriate steps
danceability = st.number_input("Danceability", min_value=0.0651, max_value=0.988, value=0.5, step=0.001)
energy = st.number_input("Energy", min_value=0.0223, max_value=0.999, value=0.5, step=0.001)
key = st.selectbox("Key", options=list(range(12)), format_func=lambda x: f"Key {x}")
loudness = st.number_input("Loudness (dB)", min_value=-22.534, max_value=1.313, value=-10.0, step=0.1)
mode = st.selectbox("Mode", options=[0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
speechiness = st.number_input("Speechiness", min_value=0.0232, max_value=0.905, value=0.1, step=0.001)
acousticness = st.number_input("Acousticness", min_value=0.000003, max_value=0.976, value=0.3, step=0.001)
instrumentalness = st.number_input("Instrumentalness", min_value=0.0, max_value=0.981, value=0.0, step=0.001)
liveness = st.number_input("Liveness", min_value=0.0205, max_value=0.988, value=0.1, step=0.001)
valence = st.number_input("Valence", min_value=0.0218, max_value=0.976, value=0.5, step=0.001)
tempo = st.number_input("Tempo (BPM)", min_value=64.934, max_value=220.138, value=120.0, step=0.1)
duration_ms = st.number_input("Duration (ms)", min_value=43807, max_value=752000, value=180000, step=1000)
time_signature = st.selectbox("Time Signature", options=[1, 2, 3, 4, 5], format_func=lambda x: f"{x}/4")

if st.button("Predict Genre"):
    # Collect features in the order expected by the model
    features = np.array([[danceability, energy, key, loudness, mode, speechiness,
                          acousticness, instrumentalness, liveness, valence,
                          tempo, duration_ms, time_signature]])
    # Scale features
    features_scaled = scaler.transform(features)
    # Predict genre and probabilities
    pred = model.predict(features_scaled)
    proba = model.predict_proba(features_scaled)[0]
    # Fix: convert pred[0] to scalar
    pred_genre = genre_map.get(pred[0].item(), f"Unknown ({pred[0].item()})")
    st.write(f"**Predicted Genre:** {pred_genre}")
    st.write("**Probability Scores:**")
    for idx, p in enumerate(proba):
        if idx in genre_map:
            st.write(f"{genre_map[idx]}: {p:.2f}")

