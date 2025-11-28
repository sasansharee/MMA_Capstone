#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np

genre_map = {
    0: "Dark Trap",
    1: "Emo",
    4: "Rap",
    6: "Trap Metal",
    7: "Underground Rap",
    8: "dnb",
    9: "hardstyle",
    10: "psytrance",
    11: "techhouse",
    12: "techno",
    14: "trap"
}

model = joblib.load(r"C:\Users\sasan\OneDrive\Desktop\streamlit\spotify_best_gradient_boosting_model.pkl")

st.title("Spotify Genre Prediction")

danceability = st.number_input(
    "Danceability", min_value=0.123, max_value=0.977, step=0.001,
    help="Range: 0.123 to 0.977"
)
energy = st.number_input(
    "Energy", min_value=0.000243, max_value=0.999, step=0.001,
    help="Range: 0.000243 to 0.999"
)
key = st.number_input(
    "Key", min_value=0, max_value=11, step=1,
    help="Range: 0 to 11"
)
loudness = st.number_input(
    "Loudness (dB)", min_value=-26.172, max_value=1.605, step=0.01,
    help="Range: -26.172 to 1.605"
)
mode = st.selectbox(
    "Mode", [0, 1],
    help="0 = Minor, 1 = Major"
)
speechiness = st.number_input(
    "Speechiness", min_value=0.0227, max_value=0.902, step=0.001,
    help="Range: 0.0227 to 0.902"
)
acousticness = st.number_input(
    "Acousticness", min_value=0.000002, max_value=0.986, step=0.001,
    help="Range: 0.000002 to 0.986"
)
instrumentalness = st.number_input(
    "Instrumentalness", min_value=0.0, max_value=0.981, step=0.001,
    help="Range: 0.0 to 0.981"
)
liveness = st.number_input(
    "Liveness", min_value=0.0202, max_value=0.978, step=0.001,
    help="Range: 0.0202 to 0.978"
)
valence = st.number_input(
    "Valence", min_value=0.0187, max_value=0.969, step=0.001,
    help="Range: 0.0187 to 0.969"
)
tempo = st.number_input(
    "Tempo", min_value=64.95, max_value=220.102, step=0.01,
    help="Range: 64.95 to 220.102"
)
duration_ms = st.number_input(
    "Duration (ms)", min_value=49227, max_value=629092, step=1000,
    help="Range: 49227 to 629092"
)
time_signature = st.selectbox(
    "Time Signature", [1, 2, 3, 4, 5],
    help="Range: 1 to 5"
)

if st.button("Predict Genre"):
    features = np.array([[danceability, energy, key, loudness, mode, speechiness, acousticness,
                          instrumentalness, liveness, valence, tempo, duration_ms, time_signature]])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    pred_genre = genre_map.get(pred, f"Unknown ({pred})")
    st.write(f"**Predicted Genre:** {pred_genre}")
    st.write("**Probability Scores:**")
    for idx, p in enumerate(proba):
        if idx in genre_map:
            st.write(f"{genre_map[idx]}: {p:.2f}")

