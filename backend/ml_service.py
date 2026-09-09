from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=150, random_state=42),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
}


@dataclass
class DatasetSummary:
    rows: int
    columns: list[str]
    preview: list[dict]


def read_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid CSV.") from exc
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("The dataset must contain data and at least two columns.")
    return frame


def summarize(frame: pd.DataFrame) -> DatasetSummary:
    preview = frame.head(5).where(pd.notna(frame.head(5)), None).to_dict(orient="records")
    return DatasetSummary(len(frame), frame.columns.astype(str).tolist(), preview)


def train_classifier(frame: pd.DataFrame, target: str, model_name: str, scale: bool) -> dict:
    if target not in frame.columns:
        raise ValueError("The selected target column does not exist.")
    if model_name not in MODELS:
        raise ValueError("Choose a supported classification model.")

    clean = frame.dropna(subset=[target]).copy()
    if len(clean) < 10:
        raise ValueError("The dataset needs at least 10 rows with a target value.")

    X = clean.drop(columns=[target])
    y = clean[target]
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformers = []
    if numeric:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("model", MODELS[model_name]()),
    ])
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, average="weighted", zero_division=0
    )
    labels = sorted(y.astype(str).unique().tolist())
    matrix = confusion_matrix(y_test.astype(str), predicted.astype(str), labels=labels)
    return {
        "model": model_name,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        },
        "confusion_matrix": {"labels": labels, "values": matrix.tolist()},
    }

