"""
Online Retail Customer Segmentation
===================================

Capstone project: Retail / Customer Clustering

This script performs:
1. Data-quality exploration
2. RFM distribution and skewness analysis
3. Model-selection analysis
4. K=2, K=3 and K=4 comparison
5. Final K=3 business profiling
6. Static visualization generation

The reusable data-processing and ML pipeline is maintained
in segmentation.py.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from segmentation import (
    load_data,
    clean_data,
    create_rfm,
    transform_rfm,
)

# Create output directory.
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. HELPER FUNCTION
# ============================================================


def save_plot(filename):
    """Save the current matplotlib figure to the outputs folder."""
    plt.tight_layout()
    plt.savefig(
        os.path.join("outputs", filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ============================================================
# 3. LOAD RAW DATA AND EXPLORATORY DATA QUALITY CHECKS
# ============================================================

# Use the shared loader from segmentation.py.
df = load_data()

print("First five rows:")
print(df.head())

print("\nShape:", df.shape)

print("\nDataFrame information:")
df.info()

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

# InvoiceDate is initially loaded as a string/object.
# Convert it to datetime so we can calculate dates
# and Recency correctly later.
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("\nInvoiceDate data type:")
print(df["InvoiceDate"].dtype)

print("\nEarliest invoice date:")
print(df["InvoiceDate"].min())

print("\nLatest invoice date:")
print(df["InvoiceDate"].max())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nQuantity statistics:")
print(df["Quantity"].describe())

print("Negative quantity:", (df["Quantity"] < 0).sum())
print("Zero quantity:", (df["Quantity"] == 0).sum())

print("\nUnitPrice statistics:")
print(df["UnitPrice"].describe())

print("Zero price:", (df["UnitPrice"] == 0).sum())
print("Negative price:", (df["UnitPrice"] < 0).sum())

cancelled_mask = df["InvoiceNo"].astype(str).str.startswith("C")

print("\nCancelled invoices:", cancelled_mask.sum())
print("\nSample cancelled invoices:")
print(df[cancelled_mask].head())


# ============================================================
# 4. CREATE A CLEAN WORKING DATASET
# ============================================================

# The reusable cleaning function is used here.
data = clean_data(df)

print("\nShape after cleaning:", data.shape)
print("\nSample cleaned transaction values:")
print(data[["Quantity", "UnitPrice", "TotalPrice"]].head())


# ============================================================
# 5. CREATE RFM FEATURES AND TRANSFORM THEM
# ============================================================

# Reuse the shared RFM creation pipeline from segmentation.py.
rfm = create_rfm(data)

print("\nFirst five RFM records:")
print(rfm.head())

print("\nRFM shape:", rfm.shape)
print("\nRFM descriptive statistics:")
print(rfm.describe())

# Reuse the shared transformation pipeline.
rfm_log, rfm_scaled = transform_rfm(rfm)

print("\nRFM after log transformation:")
print(rfm_log.head())

print("\nSkewness after log transformation:")
print(rfm_log.skew())

print("\nScaled RFM head:")
print(pd.DataFrame(rfm_scaled, columns=["Recency", "Frequency", "Monetary"]).head())


# ============================================================
# 6. RFM DISTRIBUTION ANALYSIS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(rfm["Recency"], kde=True, ax=axes[0])
axes[0].set_title("Recency Distribution")

sns.histplot(rfm["Frequency"], kde=True, ax=axes[1])
axes[1].set_title("Frequency Distribution")

sns.histplot(rfm["Monetary"], kde=True, ax=axes[2])
axes[2].set_title("Monetary Distribution")

save_plot("rfm_distributions.png")
plt.show()

print("\nRFM Skewness:")
print(rfm[["Recency", "Frequency", "Monetary"]].skew())


# ============================================================
# 7. BOX PLOT / OUTLIER ANALYSIS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(y=rfm["Recency"], ax=axes[0])
axes[0].set_title("Recency Boxplot")

sns.boxplot(y=rfm["Frequency"], ax=axes[1])
axes[1].set_title("Frequency Boxplot")

sns.boxplot(y=rfm["Monetary"], ax=axes[2])
axes[2].set_title("Monetary Boxplot")

save_plot("rfm_boxplots.png")
plt.show()


# ============================================================
# 8. ELBOW METHOD
# ============================================================

inertia = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(2, 11))
save_plot("elbow_method.png")
plt.show()


# ============================================================
# 9. SILHOUETTE ANALYSIS
# ============================================================

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    score = silhouette_score(rfm_scaled, labels)
    silhouette_scores.append(score)

print("\nSilhouette Scores:")
for k, score in zip(range(2, 11), silhouette_scores):
    print(f"K={k}: {score:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.xticks(range(2, 11))
save_plot("silhouette_scores.png")
plt.show()


# ============================================================
# 10. INITIAL K=2 MODEL
# ============================================================

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

print("\nCluster counts:")
print(rfm["Cluster"].value_counts())

print("\nCluster centers:")
print(kmeans.cluster_centers_)

cluster_summary = rfm.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean")
).round(2)

print("\nK=2 Cluster Summary:")
print(cluster_summary)


# ============================================================
# 11. PCA FOR 2-D VISUALIZATION
# ============================================================

# PCA is not used for clustering here; it is used to visualize the data.
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm["Cluster"], alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segments using PCA")
save_plot("pca_customer_segments.png")
plt.show()

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal variance explained:")
print(pca.explained_variance_ratio_.sum())

print("\nPCA Components:")
print(pd.DataFrame(
    pca.components_,
    columns=["Recency", "Frequency", "Monetary"],
    index=["PC1", "PC2"]
))


# ============================================================
# 12. COMPARE K=3 AND K=4
# ============================================================

for k in [3, 4]:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(rfm_scaled)
    rfm[f"Cluster_{k}"] = labels

for k in [3, 4]:
    print(f"\n{'=' * 50}")
    print(f"K = {k}")
    print(f"{'=' * 50}")

    summary = rfm.groupby(f"Cluster_{k}").agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean")
    ).round(2)

    print(summary)


# ============================================================
# 13. DETAILED K=3 PROFILE
# ============================================================

k3_profile = rfm.groupby("Cluster_3").agg(
    Customers=("CustomerID", "count"),
    Avg_Recency=("Recency", "mean"),
    Median_Recency=("Recency", "median"),
    Avg_Frequency=("Frequency", "mean"),
    Median_Frequency=("Frequency", "median"),
    Avg_Monetary=("Monetary", "mean"),
    Median_Monetary=("Monetary", "median")
).round(2)

print("\nK=3 Detailed Profile:")
print(k3_profile)

print(
    "\nK=3 selected for final business interpretation "
    "because it provides three interpretable behavioral groups."
)


# ============================================================
# 14. CUSTOMER AND REVENUE CONTRIBUTION
# ============================================================

segment_revenue = rfm.groupby("Cluster_3").agg(
    Customers=("CustomerID", "count"),
    Total_Revenue=("Monetary", "sum"),
    Avg_Monetary=("Monetary", "mean"),
    Median_Monetary=("Monetary", "median")
)

segment_revenue["Customer_%"] = (
    segment_revenue["Customers"] / len(rfm) * 100
)
segment_revenue["Revenue_%"] = (
    segment_revenue["Total_Revenue"] / segment_revenue["Total_Revenue"].sum() * 100
)
segment_revenue = segment_revenue.round(2)

print("\nCustomer and Revenue Contribution:")
print(segment_revenue)


# ============================================================
# 15. CUSTOMER DISTRIBUTION VISUALIZATION
# ============================================================

customer_counts = rfm["Cluster_3"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(customer_counts.index.astype(str), customer_counts.values)
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution by Segment")
save_plot("customer_distribution.png")
plt.show()


# ============================================================
# 16. REVENUE CONTRIBUTION VISUALIZATION
# ============================================================

revenue_by_segment = rfm.groupby("Cluster_3")["Monetary"].sum()

plt.figure(figsize=(8, 5))
plt.bar(revenue_by_segment.index.astype(str), revenue_by_segment.values)
plt.xlabel("Customer Segment")
plt.ylabel("Total Revenue")
plt.title("Revenue Contribution by Segment")
save_plot("revenue_contribution.png")
plt.show()


# ============================================================
# 17. RFM PROFILE VISUALIZATION
# ============================================================

cluster_profile = rfm.groupby("Cluster_3").agg(
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean")
)

print("\nAverage RFM profile:")
print(cluster_profile)

cluster_profile_scaled = (
    cluster_profile - cluster_profile.mean()
) / cluster_profile.std()

print("\nScaled cluster profile:")
print(cluster_profile_scaled.round(2))

plt.figure(figsize=(9, 6))
for cluster in cluster_profile_scaled.index:
    plt.plot(
        cluster_profile_scaled.columns,
        cluster_profile_scaled.loc[cluster],
        marker="o",
        label=f"Segment {cluster}"
    )

plt.axhline(0, linestyle="--")
plt.xlabel("RFM Metrics")
plt.ylabel("Relative Score")
plt.title("Customer Segment RFM Profile")
plt.legend()
save_plot("rfm_profile.png")
plt.show()


# ============================================================
# 18. FINAL CUSTOMER SEGMENTATION SUMMARY
# ============================================================

segment_names = {
    0: "Highly Active / High Value",
    1: "Inactive / Low Value",
    2: "Regular / Moderate Value"
}

final_summary = rfm.groupby("Cluster_3").agg(
    Customers=("CustomerID", "count"),
    Avg_Recency=("Recency", "mean"),
    Median_Recency=("Recency", "median"),
    Avg_Frequency=("Frequency", "mean"),
    Median_Frequency=("Frequency", "median"),
    Avg_Monetary=("Monetary", "mean"),
    Median_Monetary=("Monetary", "median"),
    Total_Revenue=("Monetary", "sum")
)

final_summary["Segment"] = final_summary.index.map(segment_names)
final_summary["Customer_%"] = final_summary["Customers"] / len(rfm) * 100
final_summary["Revenue_%"] = (
    final_summary["Total_Revenue"] / final_summary["Total_Revenue"].sum() * 100
)

final_summary = final_summary[
    [
        "Segment",
        "Customers",
        "Customer_%",
        "Avg_Recency",
        "Median_Recency",
        "Avg_Frequency",
        "Median_Frequency",
        "Avg_Monetary",
        "Median_Monetary",
        "Total_Revenue",
        "Revenue_%",
    ]
].round(2)

print("\n" + "=" * 70)
print("FINAL CUSTOMER SEGMENTATION SUMMARY")
print("=" * 70)
print(final_summary)

