"""
Task 1: Data Merge, EDA, and Feature Engineering
Formative 2 - Multimodal Data Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
FEATURES_DIR = "features"

def load_data():
    sp = pd.read_excel(f"{DATA_DIR}/customer_social_profiles.xlsx")
    tx = pd.read_excel(f"{DATA_DIR}/customer_transactions.xlsx")
    return sp, tx

def clean_social_profiles(sp: pd.DataFrame) -> pd.DataFrame:
    print(f"[SP] Before cleaning: {sp.shape}")
    sp = sp.drop_duplicates()
    sp["customer_id"] = sp["customer_id_new"].str.replace("A", "").astype(int)
    print(f"[SP] After dedup: {sp.shape}")
    return sp

def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    print(f"[TX] Before cleaning: {tx.shape}")
    null_rating = tx["customer_rating"].isnull().sum()
    print(f"[TX] Null customer_rating rows: {null_rating} → filling with median")
    tx["customer_rating"] = tx["customer_rating"].fillna(tx["customer_rating"].median())
    tx["customer_id"] = tx["customer_id_legacy"]
    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"])
    print(f"[TX] After cleaning: {tx.shape}")
    return tx

def merge_datasets(sp: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(tx, sp, on="customer_id", how="inner")
    print(f"\n[MERGE] Inner join result: {merged.shape}")
    print(f"[MERGE] Columns: {merged.columns.tolist()}")
    assert merged["customer_id"].isnull().sum() == 0, "Null customer IDs after merge!"
    assert merged["product_category"].isnull().sum() == 0, "Null target after merge!"
    print("[MERGE] Validation passed")
    return merged

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = df["purchase_date"].max()
    df["recency_days"] = (reference_date - df["purchase_date"]).dt.days

    df["engagement_x_interest"] = df["engagement_score"] * df["purchase_interest_score"]
    df["rating_x_amount"] = df["customer_rating"] * df["purchase_amount"]

    le_platform = LabelEncoder()
    le_sentiment = LabelEncoder()
    df["platform_encoded"] = le_platform.fit_transform(df["social_media_platform"])
    df["sentiment_encoded"] = le_sentiment.fit_transform(df["review_sentiment"])

    platform_dummies = pd.get_dummies(df["social_media_platform"], prefix="platform")
    sentiment_dummies = pd.get_dummies(df["review_sentiment"], prefix="sentiment")
    df = pd.concat([df, platform_dummies, sentiment_dummies], axis=1)

    le_target = LabelEncoder()
    df["product_label"] = le_target.fit_transform(df["product_category"])

    print(f"\n[FEATURES] Engineered dataset shape: {df.shape}")
    print(f"[FEATURES] Target classes: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    return df, le_target

def run_eda(df: pd.DataFrame):
    print("\nSummary Statistics")
    print(df[["engagement_score","purchase_interest_score","purchase_amount","customer_rating","recency_days"]].describe().round(2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Exploratory Data Analysis — Merged Customer Dataset", fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    df["product_category"].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Target: Product Category Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=30)

    ax = axes[0, 1]
    ax.hist(df["engagement_score"], bins=20, color="coral", edgecolor="white")
    ax.set_title("Engagement Score Distribution")
    ax.set_xlabel("Engagement Score")
    ax.set_ylabel("Frequency")

    ax = axes[0, 2]
    df.boxplot(column="purchase_amount", by="product_category", ax=ax)
    ax.set_title("Purchase Amount by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Amount ($)")
    plt.sca(ax)
    plt.xticks(rotation=30)

    ax = axes[1, 0]
    num_cols = ["engagement_score","purchase_interest_score","purchase_amount",
                "customer_rating","recency_days","engagement_x_interest"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")

    ax = axes[1, 1]
    df["social_media_platform"].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%",
        colors=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"])
    ax.set_title("Social Media Platform Mix")
    ax.set_ylabel("")

    ax = axes[1, 2]
    sentiment_avg = df.groupby("review_sentiment")["purchase_amount"].mean().sort_values()
    sentiment_avg.plot(kind="barh", ax=ax, color=["#C44E52","#8172B2","#55A868"])
    ax.set_title("Avg Purchase Amount by Sentiment")
    ax.set_xlabel("Avg Amount ($)")

    plt.tight_layout()
    plt.savefig("features/eda_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[EDA] Plots saved to features/eda_plots.png")

def save_merged(df: pd.DataFrame):
    df.to_csv(f"{FEATURES_DIR}/merged_dataset.csv", index=False)
    print(f"[SAVE] Merged dataset saved → {FEATURES_DIR}/merged_dataset.csv")

def main():
    import os
    os.makedirs(FEATURES_DIR, exist_ok=True)

    sp, tx = load_data()
    sp = clean_social_profiles(sp)
    tx = clean_transactions(tx)
    merged = merge_datasets(sp, tx)
    merged, le_target = engineer_features(merged)
    run_eda(merged)
    save_merged(merged)
    print("\n Task 1 complete.")
    return merged, le_target

if __name__ == "__main__":
    main()
