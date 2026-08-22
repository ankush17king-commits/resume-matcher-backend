"""
train_classifier.py

Trains a logistic regression classifier to predict "good fit" (1) vs
"poor fit" (0) for a resume-JD pair, using the engineered features
from features.py.

Why logistic regression (and not something fancier)?
- Small dataset (~1000s of rows) -> a complex model would overfit.
- Coefficients are directly interpretable: you can say exactly how
  much each feature (e.g. TF-IDF similarity, keyword overlap) pushes
  the prediction toward "good fit". Great for an interview walkthrough.
- It's the natural baseline model to reach for in this kind of
  structured-feature classification task, and a strong one at that.

Run: python3 train_classifier.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from features import FeatureExtractor, FEATURE_NAMES

DATA_PATH = "data/synthetic_dataset.csv"
MODEL_PATH = "models/classifier.joblib"
SCALER_PATH = "models/scaler.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Train / test split BEFORE fitting the TF-IDF vectorizer, so no
    # information from the test set leaks into training (a common
    # mistake worth explicitly avoiding and mentioning in interviews).
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    extractor = FeatureExtractor().fit(
        train_df["resume_text"], train_df["jd_text"]
    )

    X_train = extractor.transform(train_df["resume_text"], train_df["jd_text"])
    y_train = train_df["label"].values
    X_test = extractor.transform(test_df["resume_text"], test_df["jd_text"])
    y_test = test_df["label"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    print("\n=== Evaluation on held-out test set ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 score : {f1_score(y_test, y_pred):.3f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.3f}")

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(
        cm, index=["actual_poor_fit", "actual_good_fit"],
        columns=["pred_poor_fit", "pred_good_fit"]
    ))

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["poor_fit", "good_fit"]))

    print("\n=== Feature importance (logistic regression coefficients) ===")
    coef_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "coefficient": clf.coef_[0]
    }).sort_values("coefficient", ascending=False)
    print(coef_df.to_string(index=False))
    print(
        "\nInterpretation: positive coefficient -> higher values of that "
        "feature push the prediction toward 'good fit'. E.g. if "
        "tfidf_cosine_sim has the largest positive coefficient, overall "
        "text similarity is the strongest driver of a 'good fit' prediction."
    )

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(extractor.vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model to {MODEL_PATH}, scaler to {SCALER_PATH}, "
          f"vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
