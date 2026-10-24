"""Synthetic training data and MLP model for the PantryPulse demo."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SHELF_LIFE_DAYS = {
    "Berries": 5,
    "Leafy greens": 6,
    "Milk": 8,
    "Bread": 7,
    "Leftovers": 4,
    "Apples": 21,
    "Carrots": 28,
}

CATEGORY_FEATURES = ["food_type"]
NUMERIC_FEATURES = [
    "days_since_purchase",
    "quantity",
    "household_size",
    "servings_per_person_weekly",
]
FEATURES = CATEGORY_FEATURES + NUMERIC_FEATURES


def make_training_data(size=6000, random_state=42):
    """Create reproducible demo examples from simple shelf-life assumptions."""
    rng = np.random.default_rng(random_state)
    food_types = np.array(list(SHELF_LIFE_DAYS))
    data = pd.DataFrame(
        {
            "food_type": rng.choice(food_types, size=size),
            "days_since_purchase": rng.integers(0, 15, size=size),
            "quantity": rng.uniform(1, 12, size=size),
            "household_size": rng.integers(1, 7, size=size),
            "servings_per_person_weekly": rng.uniform(0.2, 7, size=size),
        }
    )
    shelf_life = data["food_type"].map(SHELF_LIFE_DAYS).to_numpy()
    weekly_consumption = (
        data["servings_per_person_weekly"] * data["household_size"]
    ).to_numpy()
    days_to_use = data["quantity"].to_numpy() * 7 / np.maximum(
        weekly_consumption, 0.25
    )
    risk_score = (
        data["days_since_purchase"].to_numpy() + days_to_use - shelf_life
    ) / 2.5
    risk_probability = 1 / (1 + np.exp(-risk_score))
    labels = rng.binomial(1, risk_probability)
    return data[FEATURES], labels


def train_model():
    """Fit and return the preprocessing pipeline and MLP classifier."""
    training_features, labels = make_training_data()
    preprocessing = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORY_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    max_iter=300,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(training_features, labels)
    return model


def predict_risk(model, food_type, days_since_purchase, quantity,
                 household_size, servings_per_person_weekly):
    """Return the predicted high-risk probability as a fraction between 0 and 1."""
    features = pd.DataFrame(
        [
            {
                "food_type": food_type,
                "days_since_purchase": days_since_purchase,
                "quantity": quantity,
                "household_size": household_size,
                "servings_per_person_weekly": servings_per_person_weekly,
            }
        ],
        columns=FEATURES,
    )
    return float(model.predict_proba(features)[0, 1])