import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# === Load and prepare model from your dataset ===

@st.cache_data
def load_and_train_model():
    df = pd.read_csv("vlr.csv")

    # Clean columns
    df['HS%_defend'] = df['HS%_defend'].str.replace('%', '', regex=False).astype(float)
    numeric_columns = [
        'FK_all', 'FK_attack', 'FK_defend',
        'FD_all', 'FD_attack', 'FD_defend',
        'FKFD +/-_all', 'FKFD +/-_attack', 'FKFD +/-_defend',
        'Rounds Played'
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=numeric_columns + ['HS%_defend'])
    df["Team_Won"] = (df["Team"] == df["Winner"]).astype(int)

    # Aggregate to team level
    team_stats = df.groupby(["Date", "Team", "Team_Won"]).agg({
        'FK_all': 'sum',
        'FD_all': 'sum',
        'FKFD +/-_all': 'sum',
        'HS%_defend': 'mean',
        'Rounds Played': 'mean'
    }).reset_index()

    matches = team_stats.merge(team_stats, on="Date")
    matches = matches[matches["Team_x"] != matches["Team_y"]]
    matches = matches.drop_duplicates(subset=["Date", "Team_x"])

    feature_columns = [
        "FK_all_x", "FD_all_x", "FKFD +/-_all_x", "HS%_defend_x", "Rounds Played_x",
        "FK_all_y", "FD_all_y", "FKFD +/-_all_y", "HS%_defend_y", "Rounds Played_y"
    ]
    X = matches[feature_columns]
    y = matches["Team_Won_x"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

model = load_and_train_model()

# === Streamlit UI ===
st.set_page_config("Valorant Match Predictor", layout="centered")
st.title("🎯 Valorant Match Winner Predictor")

st.markdown("Enter match stats for **Team A** and **Team B**, and get the predicted winner.")

st.subheader("📊 Team A Stats")
fk_a = st.number_input("First Kills (Team A)", value=5)
fd_a = st.number_input("First Deaths (Team A)", value=4)
plusminus_a = st.number_input("FKFD +/- (Team A)", value=1)
hs_a = st.slider("HS% (Team A)", 0.0, 100.0, 25.0)
rounds_a = st.number_input("Average Rounds Played (Team A)", value=22)

st.subheader("📊 Team B Stats")
fk_b = st.number_input("First Kills (Team B)", value=4)
fd_b = st.number_input("First Deaths (Team B)", value=5)
plusminus_b = st.number_input("FKFD +/- (Team B)", value=-1)
hs_b = st.slider("HS% (Team B)", 0.0, 100.0, 20.0)
rounds_b = st.number_input("Average Rounds Played (Team B)", value=22)

if st.button("Predict Winner"):
    input_data = np.array([[fk_a, fd_a, plusminus_a, hs_a, rounds_a,
                            fk_b, fd_b, plusminus_b, hs_b, rounds_b]])
    prediction = model.predict(input_data)[0]
    winner = "Team A" if prediction == 1 else "Team B"
    st.success(f"✅ **Predicted Winner:** {winner}")