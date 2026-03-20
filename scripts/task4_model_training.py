"""
Task 4: Model Creation and Evaluation
Formative 2 - Multimodal Data Preprocessing

Three models:
  1. Facial Recognition Model   — identifies member from image features
  2. Voiceprint Verification    — identifies member from audio features
  3. Product Recommendation     — predicts product category from merged dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, log_loss
)

FEATURES_DIR = "features"
MODELS_DIR = "models"

def evaluate_model(model, X_test, y_test, name: str, label_names=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    try:
        y_prob = model.predict_proba(X_test)
        loss = log_loss(y_test, y_prob)
    except Exception:
        loss = float("nan")

    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Log-Loss : {loss:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=label_names)}")

    return {"model": name, "accuracy": acc, "f1_weighted": f1, "log_loss": loss}

def plot_confusion_matrix(model, X_test, y_test, label_names, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=label_names,
        cmap="Blues", ax=ax, xticks_rotation=30
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"[PLOT] Confusion matrix saved → {save_path}")

def train_face_model():
    print("\nMODEL 1: Facial Recognition")
    df = pd.read_csv(f"{FEATURES_DIR}/image_features.csv")

    feature_cols = [c for c in df.columns if c.startswith("hist_") or c.startswith("stat_")]
    X = df[feature_cols].values
    y = df["member"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # few samples, so use 70/30 split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test, "Facial Recognition (Random Forest)",
                             label_names=le.classes_)
    plot_confusion_matrix(model, X_test, y_test, le.classes_,
                          "Facial Recognition — Confusion Matrix",
                          f"{FEATURES_DIR}/cm_face.png")

    joblib.dump({"model": model, "scaler": scaler, "le": le, "feature_cols": feature_cols},
                f"{MODELS_DIR}/face_model.pkl")
    print(f"[SAVE] Face model saved → {MODELS_DIR}/face_model.pkl")
    return metrics

def train_voice_model():
    print("\nMODEL 2: Voiceprint Verification")
    audio_csv = f"{FEATURES_DIR}/audio_features.csv"
    if not os.path.exists(audio_csv):
        print(f"  [SKIP] {audio_csv} not found — run task3_audio_processing.py first")
        print("         (requires: pip install librosa soundfile)")
        return None
    df = pd.read_csv(audio_csv)

    feature_cols = [c for c in df.columns if c.startswith("mfcc_") or
                    c in ["spectral_rolloff_mean","spectral_rolloff_std",
                           "rms_energy_mean","rms_energy_std","zcr_mean","spectral_centroid_mean"]]
    X = df[feature_cols].values
    y = df["member"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test, "Voiceprint Verification (Random Forest)",
                             label_names=le.classes_)
    plot_confusion_matrix(model, X_test, y_test, le.classes_,
                          "Voiceprint Verification — Confusion Matrix",
                          f"{FEATURES_DIR}/cm_voice.png")

    joblib.dump({"model": model, "scaler": scaler, "le": le, "feature_cols": feature_cols},
                f"{MODELS_DIR}/voice_model.pkl")
    print(f"[SAVE] Voice model saved → {MODELS_DIR}/voice_model.pkl")
    return metrics

def train_product_model():
    print("\nMODEL 3: Product Recommendation")
    df = pd.read_csv(f"{FEATURES_DIR}/merged_dataset.csv")

    feature_cols = [
        "engagement_score", "purchase_interest_score", "purchase_amount",
        "customer_rating", "recency_days", "engagement_x_interest", "rating_x_amount",
        "platform_encoded", "sentiment_encoded",
    ]
    dummy_cols = [c for c in df.columns if c.startswith("platform_") or c.startswith("sentiment_")]
    feature_cols += dummy_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df["product_label"].values
    product_names = df["product_category"].unique()

    le = LabelEncoder()
    le.fit(df["product_category"])
    label_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # pick the best by 5-fold CV accuracy
    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_model, best_score = None, None, -1

    print("  Cross-validation (5-fold):")
    for name, clf in candidates.items():
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        scores = cross_val_score(clf, X_tr_s, y_train, cv=cv, scoring="accuracy")
        print(f"    {name}: {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name
            best_model = clf
            best_scaler = scaler

    print(f"\n  [PASS] Best model: {best_name} (CV acc={best_score:.4f})")
    best_scaler.fit(X_train)
    best_model.fit(best_scaler.transform(X_train), y_train)

    metrics = evaluate_model(best_model, best_scaler.transform(X_test), y_test,
                             f"Product Recommendation ({best_name})",
                             label_names=label_names)
    plot_confusion_matrix(best_model, best_scaler.transform(X_test), y_test,
                          label_names, "Product Recommendation — Confusion Matrix",
                          f"{FEATURES_DIR}/cm_product.png")

    joblib.dump({"model": best_model, "scaler": best_scaler, "le": le, "feature_cols": feature_cols},
                f"{MODELS_DIR}/product_model.pkl")
    print(f"[SAVE] Product model saved → {MODELS_DIR}/product_model.pkl")
    return metrics, feature_cols

def plot_metrics_summary(metrics_list: list):
    df_m = pd.DataFrame([m for m in metrics_list if isinstance(m, dict)])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Model Performance Summary", fontsize=13, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, metric in zip(axes, ["accuracy", "f1_weighted"]):
        bars = ax.bar(df_m["model"], df_m[metric], color=colors)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis='x', rotation=20)
        for bar, val in zip(bars, df_m[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FEATURES_DIR}/model_metrics_summary.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[PLOT] Metrics summary saved → features/model_metrics_summary.png")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    face_metrics = train_face_model()
    voice_metrics = train_voice_model()
    product_result = train_product_model()
    product_metrics = product_result[0] if isinstance(product_result, tuple) else product_result

    all_metrics = [m for m in [face_metrics, voice_metrics, product_metrics] if m is not None]
    plot_metrics_summary(all_metrics)
    print("\n Task 4 complete — all models trained and saved.")

if __name__ == "__main__":
    main()
