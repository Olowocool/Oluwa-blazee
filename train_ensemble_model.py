import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_ensemble_model():

    df = pd.read_csv("data/learning_dataset.csv")

    df = df.dropna()

    X = df.drop(columns=["result"])
    y = df["result"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, "models/ensemble_model.joblib")

    print("Ensemble model trained successfully.")
