"""
Online Retail Customer Segmentation
===================================

Capstone project: Retail / Customer Clustering

Overall workflow:
1. Load transaction data from SQLite
2. Explore data quality
3. Clean the transaction data
4. Build customer-level RFM features
5. Analyze skewness and distributions
6. Apply log transformation
7. Standardize RFM features
8. Evaluate possible values of K
9. Compare K=2, K=3 and K=4
10. Use PCA for visualization
11. Profile the final K=3 segments
12. Calculate customer and revenue contribution
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import sqlite3                    # Connect to the SQLite database
import pandas as pd               # Data loading, cleaning and analysis
import numpy as np                # Numerical operations, including log1p
import matplotlib.pyplot as plt   # Plotting
import seaborn as sns              # Statistical visualizations

# Scikit-learn libraries used later in the project
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import os

# Create an outputs folder if it does not already exist.
os.makedirs("outputs", exist_ok=True)

# ============================================================
# 2. CONNECT TO THE SQLITE DATABASE
# ============================================================

# Connect Python to the SQLite database.
# The database file must be in the current working directory.
conn = sqlite3.connect("clustering.db")


# ============================================================
# 3. CHECK AVAILABLE TABLES
# ============================================================

# SQLite stores information about all tables in sqlite_master.
# This query helps us verify which datasets are available
# before selecting the dataset required for the project.
query = """
SELECT name
FROM sqlite_master
WHERE type='table';
"""

tables = pd.read_sql_query(query, conn)

print("Available tables:")
print(tables)


# ============================================================
# 4. LOAD THE ONLINE RETAIL CUSTOMER DATA
# ============================================================

# Load the complete online_retail_customers table into a Pandas
# DataFrame so that we can perform exploration and preprocessing.
df = pd.read_sql_query(
    "SELECT * FROM online_retail_customers",
    conn
)

print("\nFirst five rows:")
print(df.head())

print("\nShape:", df.shape)

# info() already prints its output, so we do not need
# to wrap it inside print().
print("\nDataFrame information:")
df.info()

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 5. INITIAL DATA EXPLORATION
# ============================================================

# InvoiceDate is initially loaded as a string/object.
# Convert it to datetime so that we can calculate dates
# and Recency correctly later.
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("\nInvoiceDate data type:")
print(df["InvoiceDate"].dtype)

print("\nEarliest invoice date:")
print(df["InvoiceDate"].min())

print("\nLatest invoice date:")
print(df["InvoiceDate"].max())


# ------------------------------------------------------------
# Check duplicate transactions
# ------------------------------------------------------------

# duplicated() identifies rows where all column values are
# repeated. These are useful to identify during data-quality
# analysis before modeling.
print("\nDuplicate rows:", df.duplicated().sum())


# ------------------------------------------------------------
# Check Quantity
# ------------------------------------------------------------

# describe() gives basic statistics such as count, mean,
# minimum, maximum and quartiles.
print("\nQuantity statistics:")
print(df["Quantity"].describe())

# Negative quantities can indicate returns/cancellations
# in this dataset, so we inspect their frequency.
print("Negative quantity:", (df["Quantity"] < 0).sum())

print("Zero quantity:", (df["Quantity"] == 0).sum())


# ------------------------------------------------------------
# Check UnitPrice
# ------------------------------------------------------------

print("\nUnitPrice statistics:")
print(df["UnitPrice"].describe())

print("Zero price:", (df["UnitPrice"] == 0).sum())

print("Negative price:", (df["UnitPrice"] < 0).sum())


# ------------------------------------------------------------
# Check cancelled invoices
# ------------------------------------------------------------

# In this dataset, cancelled invoices are identified by
# InvoiceNo values beginning with "C".
cancelled_mask = df["InvoiceNo"].astype(str).str.startswith("C")

print(
    "Cancelled invoices:",
    cancelled_mask.sum()
)

# Display a few cancelled transactions so that we can
# visually confirm what these records look like.
print("\nSample cancelled invoices:")
print(df[cancelled_mask].head())


# ============================================================
# 6. CREATE A CLEAN WORKING DATASET
# ============================================================

# Keep the original df unchanged.
# All preprocessing will be performed on a copy called data.
data = df.copy()


# ------------------------------------------------------------
# Remove transactions without CustomerID
# ------------------------------------------------------------

# RFM is calculated at customer level.
# Without CustomerID, a transaction cannot be assigned
# to a specific customer, so these rows are removed.
data = data.dropna(subset=["CustomerID"])

print(
    "\nShape after removing missing CustomerID:",
    data.shape
)


# ------------------------------------------------------------
# Ensure InvoiceDate is datetime
# ------------------------------------------------------------

# This is technically already converted above, but keeping
# this line here makes the cleaning stage self-contained.
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

print("InvoiceDate data type:", data["InvoiceDate"].dtype)


# ------------------------------------------------------------
# Remove cancelled orders
# ------------------------------------------------------------

# Cancelled invoices do not represent completed purchases,
# so they are excluded from the customer purchase analysis.
data = data[
    ~data["InvoiceNo"].astype(str).str.startswith("C")
]

print(
    "Shape after removing cancelled orders:",
    data.shape
)


# ------------------------------------------------------------
# Create TotalPrice
# ------------------------------------------------------------

# Monetary value for each transaction is calculated as:
#
#       Quantity × UnitPrice
#
# This transaction-level value will later be summed for each
# customer to calculate the Monetary RFM metric.
data["TotalPrice"] = (
    data["Quantity"] * data["UnitPrice"]
)

print("\nSample transaction values:")
print(
    data[
        ["Quantity", "UnitPrice", "TotalPrice"]
    ].head()
)


# ============================================================
# 7. CREATE CUSTOMER-LEVEL RFM FEATURES
# ============================================================

# The latest transaction date in the cleaned dataset is used
# as the basis for calculating customer Recency.
latest_date = data["InvoiceDate"].max()

print("\nLatest invoice date:", latest_date)


# Add one day so that the latest customer's purchase receives
# a Recency value of 1 instead of 0.
reference_date = latest_date + pd.Timedelta(days=1)

print("Reference date:", reference_date)


# ------------------------------------------------------------
# RFM definitions
# ------------------------------------------------------------
#
# Recency:
#   Number of days since the customer's most recent purchase.
#   Lower = more recent customer activity.
#
# Frequency:
#   Number of unique invoices/orders placed by the customer.
#   Higher = more frequent purchasing.
#
# Monetary:
#   Total transaction value generated by the customer.
#   Higher = greater monetary contribution.
#
# groupby("CustomerID") converts transaction-level data into
# one row per customer.
rfm = data.groupby("CustomerID").agg(
    Recency=(
        "InvoiceDate",
        lambda x: (reference_date - x.max()).days
    ),

    Frequency=(
        "InvoiceNo",
        "nunique"
    ),

    Monetary=(
        "TotalPrice",
        "sum"
    )
).reset_index()


print("\nFirst five RFM records:")
print(rfm.head())

print("\nRFM shape:", rfm.shape)

print("\nCleaned transaction data shape:", data.shape)

print("\nRFM descriptive statistics:")
print(rfm.describe())


# ============================================================
# 8. RFM DISTRIBUTION ANALYSIS
# ============================================================

# Histograms help us understand how customers are distributed
# across Recency, Frequency and Monetary values.
#
# We expect Frequency and Monetary to be highly right-skewed
# because a small number of customers may purchase very often
# or spend much more than the typical customer.
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5)
)

sns.histplot(
    rfm["Recency"],
    kde=True,
    ax=axes[0]
)
axes[0].set_title("Recency Distribution")

sns.histplot(
    rfm["Frequency"],
    kde=True,
    ax=axes[1]
)
axes[1].set_title("Frequency Distribution")

sns.histplot(
    rfm["Monetary"],
    kde=True,
    ax=axes[2]
)
axes[2].set_title("Monetary Distribution")

plt.tight_layout()
plt.savefig("outputs/rfm_distributions.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# Measure skewness
# ------------------------------------------------------------

# Skewness measures how asymmetric a distribution is.
# A large positive value indicates a long right tail.
print("\nRFM Skewness:")
print(
    rfm[
        ["Recency", "Frequency", "Monetary"]
    ].skew()
)


# ============================================================
# 9. BOX PLOT / OUTLIER ANALYSIS
# ============================================================

# Boxplots show:
# - Median
# - Quartiles
# - Overall spread
# - Potential extreme observations
#
# An outlier is not automatically an invalid record.
# In retail data, a high-value customer can be a legitimate
# and important business observation.
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5)
)

sns.boxplot(
    y=rfm["Recency"],
    ax=axes[0]
)
axes[0].set_title("Recency Boxplot")

sns.boxplot(
    y=rfm["Frequency"],
    ax=axes[1]
)
axes[1].set_title("Frequency Boxplot")

sns.boxplot(
    y=rfm["Monetary"],
    ax=axes[2]
)
axes[2].set_title("Monetary Boxplot")

plt.tight_layout()
plt.savefig("outputs/rfm_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 10. LOG TRANSFORMATION
# ============================================================

# Frequency and Monetary are strongly right-skewed.
# K-Means is sensitive to scale and extreme values, so we
# reduce the effect of the long right tail before clustering.
#
# log1p(x) = log(1 + x)
#
# We use log1p instead of log(x) because Monetary can contain
# zero values, and log(0) is undefined.
#
# The transformation compresses very large values without
# deleting legitimate high-value customers.
rfm_log = rfm[
    ["Recency", "Frequency", "Monetary"]
].copy()

rfm_log = np.log1p(rfm_log)

print("\nRFM after log transformation:")
print(rfm_log.head())

print("\nSkewness after log transformation:")
print(rfm_log.skew())


# ============================================================
# 11. STANDARDIZE THE RFM FEATURES
# ============================================================

# K-Means calculates distances between observations.
# Recency, Frequency and Monetary have different numerical
# ranges, so we standardize them to comparable scales.
#
# StandardScaler transforms each feature approximately to:
#     mean = 0
#     standard deviation = 1
scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_log)

# Convert the NumPy array back to a DataFrame so that the
# feature names remain visible during analysis.
rfm_scaled = pd.DataFrame(
    rfm_scaled,
    columns=[
        "Recency",
        "Frequency",
        "Monetary"
    ]
)

print("\nScaled RFM:")
print(rfm_scaled.head())

print("\nScaled RFM mean:")
print(rfm_scaled.mean())

print("\nScaled RFM standard deviation:")
print(rfm_scaled.std())


# ============================================================
# 12. ELBOW METHOD
# ============================================================

# The Elbow Method evaluates different values of K.
#
# Inertia measures the total within-cluster squared distance.
# Lower inertia means the observations are closer to their
# assigned cluster centers.
#
# Inertia always decreases as K increases, so we look for
# a noticeable "bend" rather than simply choosing the lowest
# inertia value.
inertia = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(rfm_scaled)

    inertia.append(
        kmeans.inertia_
    )


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(2, 11))
plt.savefig("outputs/elbow_method.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 13. SILHOUETTE ANALYSIS
# ============================================================

# Silhouette score evaluates how well-separated the clusters
# are.
#
# Approximate interpretation:
#   Higher score = better separation/cohesion
#   Lower score  = more overlap between clusters
#
# We calculate it for K=2 through K=10.
silhouette_scores = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        rfm_scaled
    )

    score = silhouette_score(
        rfm_scaled,
        labels
    )

    silhouette_scores.append(score)


print("\nSilhouette Scores:")

for k, score in zip(
    range(2, 11),
    silhouette_scores
):
    print(
        f"K={k}: {score:.4f}"
    )


# Visualize silhouette scores to compare K values.
plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.xticks(range(2, 11))
plt.tight_layout()
plt.savefig("outputs/silhouette_scores.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 14. INITIAL K=2 MODEL
# ============================================================

# Silhouette analysis showed that K=2 produced the highest
# numerical silhouette score.
#
# We keep this model as an important validation result.
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    rfm_scaled
)

print("\nCluster counts:")
print(
    rfm["Cluster"].value_counts()
)

print("\nCluster centers:")
print(
    kmeans.cluster_centers_
)


# Summarize the behavioral characteristics of each K=2 cluster.
cluster_summary = rfm.groupby("Cluster").agg(
    Customers=(
        "CustomerID",
        "count"
    ),

    Avg_Recency=(
        "Recency",
        "mean"
    ),

    Avg_Frequency=(
        "Frequency",
        "mean"
    ),

    Avg_Monetary=(
        "Monetary",
        "mean"
    )
).round(2)

print("\nK=2 Cluster Summary:")
print(cluster_summary)


# ============================================================
# 15. PCA FOR 2-D VISUALIZATION
# ============================================================

# PCA is NOT used to create the clusters.
#
# Its purpose here is visualization: it reduces the three
# standardized RFM dimensions to two principal components so
# that we can plot the customer groups in 2-D.
pca = PCA(
    n_components=2
)

rfm_pca = pca.fit_transform(
    rfm_scaled
)


plt.figure(figsize=(8, 6))

plt.scatter(
    rfm_pca[:, 0],
    rfm_pca[:, 1],
    c=rfm["Cluster"],
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segments using PCA")
plt.tight_layout()
plt.savefig("outputs/pca_customer_segments.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# PCA explained variance
# ------------------------------------------------------------

# Explained variance tells us how much of the original
# information is represented by each principal component.
print("\nExplained variance ratio:")
print(
    pca.explained_variance_ratio_
)

print("\nTotal variance explained:")
print(
    pca.explained_variance_ratio_.sum()
)


# ------------------------------------------------------------
# PCA component loadings
# ------------------------------------------------------------

# Component loadings show how strongly each original RFM
# feature contributes to each principal component.
print("\nPCA Components:")

print(
    pd.DataFrame(
        pca.components_,
        columns=[
            "Recency",
            "Frequency",
            "Monetary"
        ],
        index=[
            "PC1",
            "PC2"
        ]
    )
)


# ============================================================
# 16. COMPARE K=3 AND K=4
# ============================================================

# K=2 has the highest silhouette score, but we also inspect
# K=3 and K=4 because additional clusters can provide more
# granular and business-interpretable customer groups.
for k in [3, 4]:

    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans_test.fit_predict(
        rfm_scaled
    )

    # Store each model's labels separately so that we can
    # compare K=3 and K=4 without overwriting the results.
    rfm[f"Cluster_{k}"] = labels


# Print a behavioral summary for K=3 and K=4.
for k in [3, 4]:

    print(f"\n{'=' * 50}")
    print(f"K = {k}")
    print(f"{'=' * 50}")

    summary = rfm.groupby(
        f"Cluster_{k}"
    ).agg(
        Customers=(
            "CustomerID",
            "count"
        ),

        Avg_Recency=(
            "Recency",
            "mean"
        ),

        Avg_Frequency=(
            "Frequency",
            "mean"
        ),

        Avg_Monetary=(
            "Monetary",
            "mean"
        )
    ).round(2)

    print(summary)


# ============================================================
# 17. DETAILED K=3 PROFILE
# ============================================================

# K=3 is used for the final business interpretation.
#
# We calculate both mean and median values because RFM data
# can contain extreme values. Median gives a more robust view
# of the typical customer in each segment.
k3_profile = rfm.groupby(
    "Cluster_3"
).agg(
    Customers=(
        "CustomerID",
        "count"
    ),

    Avg_Recency=(
        "Recency",
        "mean"
    ),

    Median_Recency=(
        "Recency",
        "median"
    ),

    Avg_Frequency=(
        "Frequency",
        "mean"
    ),

    Median_Frequency=(
        "Frequency",
        "median"
    ),

    Avg_Monetary=(
        "Monetary",
        "mean"
    ),

    Median_Monetary=(
        "Monetary",
        "median"
    )
).round(2)

print("\nK=3 Detailed Profile:")
print(k3_profile)


# ============================================================
# 18. WHY K=3 IS USED FOR BUSINESS INTERPRETATION
# ============================================================

# IMPORTANT:
#
# Silhouette score identified K=2 as the strongest separation
# according to the mathematical metric.
#
# However, K=3 provides three more granular and interpretable
# customer groups:
#
#   Segment 0 -> Highly Active / High Value
#   Segment 1 -> Inactive / Low Value
#   Segment 2 -> Regular / Moderate Value
#
# Therefore, K=3 is selected as the final BUSINESS
# segmentation while K=2 remains the strongest result according
# to silhouette score.
#
# This distinction should be mentioned in the capstone/viva.

print(
    "\nK=3 selected for final business interpretation "
    "because it provides three interpretable behavioral groups."
)


# ============================================================
# 19. CUSTOMER AND REVENUE CONTRIBUTION
# ============================================================

# This analysis answers an important business question:
#
# "How many customers are in each segment, and how much total
# monetary value does each segment contribute?"
segment_revenue = rfm.groupby(
    "Cluster_3"
).agg(
    Customers=(
        "CustomerID",
        "count"
    ),

    Total_Revenue=(
        "Monetary",
        "sum"
    ),

    Avg_Monetary=(
        "Monetary",
        "mean"
    ),

    Median_Monetary=(
        "Monetary",
        "median"
    )
)


# Percentage of the total customer base represented by
# each segment.
segment_revenue["Customer_%"] = (
    segment_revenue["Customers"]
    / len(rfm)
    * 100
)


# Percentage of total monetary value contributed by
# each segment.
segment_revenue["Revenue_%"] = (
    segment_revenue["Total_Revenue"]
    / segment_revenue["Total_Revenue"].sum()
    * 100
)

segment_revenue = segment_revenue.round(2)

print("\nCustomer and Revenue Contribution:")
print(segment_revenue)


# ============================================================
# 20. CUSTOMER DISTRIBUTION VISUALIZATION
# ============================================================

# Count how many customers belong to each K=3 segment.
customer_counts = (
    rfm["Cluster_3"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))

plt.bar(
    customer_counts.index.astype(str),
    customer_counts.values
)

plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution by Segment")
plt.tight_layout()
plt.savefig("outputs/customer_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 21. REVENUE CONTRIBUTION VISUALIZATION
# ============================================================

# Sum Monetary value for each customer segment.
revenue_by_segment = (
    rfm.groupby("Cluster_3")["Monetary"]
    .sum()
)

plt.figure(figsize=(8, 5))

plt.bar(
    revenue_by_segment.index.astype(str),
    revenue_by_segment.values
)

plt.xlabel("Customer Segment")
plt.ylabel("Total Revenue")
plt.title("Revenue Contribution by Segment")
plt.tight_layout()
plt.savefig("outputs/revenue_contribution.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 22. RFM PROFILE VISUALIZATION
# ============================================================

# Calculate the average RFM values for each K=3 segment.
cluster_profile = rfm.groupby(
    "Cluster_3"
).agg(
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean")
)

print("\nAverage RFM profile:")
print(cluster_profile)


# ------------------------------------------------------------
# Standardize the cluster averages ONLY for visualization
# ------------------------------------------------------------
#
# This is NOT another clustering transformation.
# It simply puts the three average RFM values on a comparable
# relative scale so that their patterns can be plotted together.
cluster_profile_scaled = (
    cluster_profile
    - cluster_profile.mean()
) / cluster_profile.std()

print("\nScaled cluster profile:")
print(
    cluster_profile_scaled.round(2)
)


plt.figure(figsize=(9, 6))

for cluster in cluster_profile_scaled.index:

    plt.plot(
        cluster_profile_scaled.columns,
        cluster_profile_scaled.loc[cluster],
        marker="o",
        label=f"Segment {cluster}"
    )

# Zero represents the average across the three cluster profiles.
plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("RFM Metrics")
plt.ylabel("Relative Score")
plt.title("Customer Segment RFM Profile")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/rfm_profile.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 23. FINAL CUSTOMER SEGMENTATION SUMMARY
# ============================================================

# Give the numerical cluster IDs meaningful business names.
#
# IMPORTANT:
# These names are based on the observed RFM characteristics.
# They are not produced automatically by K-Means.
segment_names = {
    0: "Highly Active / High Value",
    1: "Inactive / Low Value",
    2: "Regular / Moderate Value"
}


# Build one final table containing all important segment metrics.
final_summary = rfm.groupby(
    "Cluster_3"
).agg(
    Customers=(
        "CustomerID",
        "count"
    ),

    Avg_Recency=(
        "Recency",
        "mean"
    ),

    Median_Recency=(
        "Recency",
        "median"
    ),

    Avg_Frequency=(
        "Frequency",
        "mean"
    ),

    Median_Frequency=(
        "Frequency",
        "median"
    ),

    Avg_Monetary=(
        "Monetary",
        "mean"
    ),

    Median_Monetary=(
        "Monetary",
        "median"
    ),

    Total_Revenue=(
        "Monetary",
        "sum"
    )
)


# Convert cluster IDs into human-readable segment names.
final_summary["Segment"] = (
    final_summary.index.map(segment_names)
)


# Calculate the percentage of customers in each segment.
final_summary["Customer_%"] = (
    final_summary["Customers"]
    / len(rfm)
    * 100
)


# Calculate the percentage of total monetary value generated
# by each segment.
final_summary["Revenue_%"] = (
    final_summary["Total_Revenue"]
    / final_summary["Total_Revenue"].sum()
    * 100
)


# Reorder columns so that the final report is easier to read.
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
        "Revenue_%"
    ]
].round(2)


# Final result of the entire segmentation pipeline.
print("\n" + "=" * 70)
print("FINAL CUSTOMER SEGMENTATION SUMMARY")
print("=" * 70)
print(final_summary)


# ============================================================
# 24. CLOSE THE DATABASE CONNECTION
# ============================================================

# We are finished reading from the SQLite database, so we can
# explicitly close the connection.
conn.close()
