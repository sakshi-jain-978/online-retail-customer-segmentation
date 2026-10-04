
"""
Reusable customer segmentation pipeline.

This module contains the common data loading,
cleaning, RFM creation, transformation, K-Means
clustering, and PCA logic used by the project.

The Streamlit dashboard and analysis script can
reuse these functions instead of duplicating the
machine learning logic.
"""

# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import sqlite3

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================================================
# 2. DATABASE CONFIGURATION
# =========================================================

DB_PATH = "clustering.db"


# =========================================================
# 3. LOAD DATA
# =========================================================

def load_data(db_path=DB_PATH):
    """
    Load the online retail dataset from SQLite.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    pandas.DataFrame
        Raw online retail transaction data.
    """

    conn = sqlite3.connect(db_path)

    query = """
        SELECT *
        FROM online_retail_customers
    """

    data = pd.read_sql_query(query, conn)

    conn.close()

    return data


# =========================================================
# 4. CLEAN TRANSACTION DATA
# =========================================================

def clean_data(df):
    """
    Clean the transaction-level retail data.

    Cleaning steps:
    - Remove rows without CustomerID.
    - Convert InvoiceDate to datetime.
    - Remove cancelled invoices.
    - Calculate transaction-level TotalPrice.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw transaction data.

    Returns
    -------
    pandas.DataFrame
        Cleaned transaction data.
    """

    data = df.copy()

    # Customer-level segmentation requires a CustomerID.
    data = data.dropna(subset=["CustomerID"])

    # Convert date column to datetime.
    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"]
    )

    # Remove cancelled invoices.
    data = data[
        ~data["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]

    # Calculate transaction value.
    data["TotalPrice"] = (
        data["Quantity"] * data["UnitPrice"]
    )

    return data


# =========================================================
# 5. CREATE RFM FEATURES
# =========================================================

def create_rfm(data):
    """
    Create customer-level RFM features.

    RFM:
    - Recency: How recently the customer purchased.
    - Frequency: Number of unique invoices.
    - Monetary: Total customer monetary value.

    Parameters
    ----------
    data : pandas.DataFrame
        Cleaned transaction data.

    Returns
    -------
    pandas.DataFrame
        Customer-level RFM dataframe.
    """

    latest_date = data["InvoiceDate"].max()

    # Add one day so the latest customer has Recency = 1.
    reference_date = (
        latest_date + pd.Timedelta(days=1)
    )

    rfm = (
        data.groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x:
                (reference_date - x.max()).days
            ),
            Frequency=(
                "InvoiceNo",
                "nunique"
            ),
            Monetary=(
                "TotalPrice",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm


# =========================================================
# 6. TRANSFORM AND SCALE RFM
# =========================================================

def transform_rfm(rfm):
    """
    Apply log transformation and standardization.

    Log transformation reduces the impact of highly
    skewed RFM values.

    StandardScaler puts all three RFM variables on
    a comparable numerical scale.

    Parameters
    ----------
    rfm : pandas.DataFrame
        Customer-level RFM dataframe.

    Returns
    -------
    tuple
        rfm_log, rfm_scaled
    """

    rfm_features = rfm[
        [
            "Recency",
            "Frequency",
            "Monetary"
        ]
    ].copy()

    # Reduce skewness using log1p(x) = log(1 + x).
    rfm_log = np.log1p(rfm_features)

    # Standardize the transformed features.
    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm_log
    )

    return rfm_log, rfm_scaled


# =========================================================
# 7. APPLY K-MEANS CLUSTERING
# =========================================================

def apply_kmeans(rfm, rfm_scaled, n_clusters=3):
    """
    Apply K-Means clustering to standardized RFM data.

    Parameters
    ----------
    rfm : pandas.DataFrame
        Customer-level RFM dataframe.
    rfm_scaled : numpy.ndarray
        Standardized RFM features.
    n_clusters : int
        Number of K-Means clusters.

    Returns
    -------
    pandas.DataFrame
        RFM dataframe with cluster assignments.
    """

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    rfm = rfm.copy()

    rfm["Cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    return rfm


# =========================================================
# 8. ASSIGN BUSINESS SEGMENT NAMES
# =========================================================

def assign_segment_names(rfm):
    """
    Convert K-Means cluster numbers into business labels.

    The cluster numbers themselves have no business meaning,
    so descriptive labels make the output easier to interpret.

    Parameters
    ----------
    rfm : pandas.DataFrame
        RFM dataframe containing Cluster.

    Returns
    -------
    pandas.DataFrame
        RFM dataframe with Segment column.
    """

    segment_names = {
        0: "Highly Active / High Value",
        1: "Inactive / Low Value",
        2: "Regular / Moderate Value"
    }

    rfm = rfm.copy()

    rfm["Segment"] = (
        rfm["Cluster"]
        .map(segment_names)
    )

    return rfm


# =========================================================
# 9. PCA FOR VISUALIZATION
# =========================================================

def apply_pca(rfm, rfm_scaled):
    """
    Apply PCA for two-dimensional visualization.

    PCA is NOT used for clustering.
    It is used only to visualize the customer segments.

    Parameters
    ----------
    rfm : pandas.DataFrame
        RFM dataframe.
    rfm_scaled : numpy.ndarray
        Standardized RFM features.

    Returns
    -------
    tuple
        Updated RFM dataframe and explained variance.
    """

    pca = PCA(n_components=2)

    rfm_pca = pca.fit_transform(
        rfm_scaled
    )

    rfm = rfm.copy()

    rfm["PC1"] = rfm_pca[:, 0]
    rfm["PC2"] = rfm_pca[:, 1]

    explained_variance = (
        pca.explained_variance_ratio_ * 100
    )

    return rfm, explained_variance


# =========================================================
# 10. COMPLETE PIPELINE
# =========================================================

def run_segmentation(db_path=DB_PATH, n_clusters=3):
    """
    Run the complete customer segmentation pipeline.

    Flow:
        Database
        → Cleaning
        → RFM
        → Log transformation
        → Standardization
        → K-Means
        → Segment names
        → PCA

    Returns
    -------
    tuple
        rfm, explained_variance
    """

    # Load raw transaction data.
    df = load_data(db_path)

    # Clean transaction data.
    data = clean_data(df)

    # Create customer-level RFM features.
    rfm = create_rfm(data)

    # Transform and scale RFM features.
    _, rfm_scaled = transform_rfm(rfm)

    # Apply K-Means clustering.
    rfm = apply_kmeans(
        rfm,
        rfm_scaled,
        n_clusters=n_clusters
    )

    # Add business-friendly segment names.
    rfm = assign_segment_names(rfm)

    # Apply PCA for visualization.
    rfm, explained_variance = apply_pca(
        rfm,
        rfm_scaled
    )

    return rfm, explained_variance