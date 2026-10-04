"""
Online Retail Customer Segmentation - Streamlit Dashboard

This application:
1. Loads the online retail dataset from SQLite.
2. Cleans the transaction data.
3. Creates customer-level RFM features.
4. Applies log transformation.
5. Standardizes the RFM features.
6. Applies K-Means clustering with K=3.
7. Displays customer segments and business insights.
8. Provides interactive customer lookup.
9. Visualizes the clusters using PCA.
"""

# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import pandas as pd
import streamlit as st
import plotly.express as px

from segmentation import run_segmentation


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Online Retail Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)


# =========================================================
# 3. APPLICATION TITLE
# =========================================================

st.title("🛒 Online Retail Customer Segmentation")

st.markdown(
    """
    **RFM Analysis + K-Means Clustering**

    An interactive dashboard for understanding customer
    behavior, value, and actionable customer segments.
    """
)

st.divider()


# =========================================================
# LOAD SEGMENTATION RESULTS
# =========================================================


@st.cache_data
def get_segmentation_results():
    return run_segmentation(n_clusters=3)

rfm, explained_variance = get_segmentation_results()
# =========================================================
# 4. BUSINESS METRICS
# =========================================================

total_customers = len(rfm)

total_monetary = rfm["Monetary"].sum()

average_customer_value = (
    rfm["Monetary"].mean()
)

number_of_segments = (
    rfm["Segment"].nunique()
)


# =========================================================
# 5. KPI CARDS
# =========================================================

st.subheader("📊 Business Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with col2:

    st.metric(
        "Total Customer Value",
        f"₹{total_monetary:,.0f}"
    )


with col3:

    st.metric(
        "Average Customer Value",
        f"₹{average_customer_value:,.0f}"
    )


with col4:

    st.metric(
        "Customer Segments",
        number_of_segments
    )


st.divider()


# =========================================================
# 6. SEGMENT SUMMARY
# =========================================================

st.subheader("🎯 Customer Segment Overview")


segment_summary = (
    rfm.groupby("Segment")
    .agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Monetary=("Monetary", "sum")
    )
    .reset_index()
)


segment_summary["Customer_%"] = (
    segment_summary["Customers"]
    / total_customers
    * 100
)


segment_summary["Revenue_%"] = (
    segment_summary["Total_Monetary"]
    / total_monetary
    * 100
)


segment_summary = segment_summary.round(2)


# Display only the business-friendly columns.

display_summary = segment_summary[
    [
        "Segment",
        "Customers",
        "Customer_%",
        "Avg_Recency",
        "Avg_Frequency",
        "Avg_Monetary",
        "Revenue_%"
    ]
]


st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 11. CUSTOMER DISTRIBUTION + REVENUE CONTRIBUTION
# =========================================================

st.subheader("📈 Customer and Revenue Distribution")


chart_col1, chart_col2 = st.columns(2)


# ---------------------------------------------------------
# Customer distribution
# ---------------------------------------------------------

with chart_col1:

    customer_chart = px.bar(
        segment_summary,
        x="Segment",
        y="Customer_%",
        title="Customer Distribution (%)",
        text="Customer_%",
        labels={
            "Customer_%": "Customers (%)"
        }
    )

    customer_chart.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    customer_chart.update_layout(
        xaxis_title="",
        yaxis_title="Customer Share (%)"
    )

    st.plotly_chart(
        customer_chart,
        use_container_width=True
    )


# ---------------------------------------------------------
# Revenue contribution
# ---------------------------------------------------------

with chart_col2:

    revenue_chart = px.bar(
        segment_summary,
        x="Segment",
        y="Revenue_%",
        title="Revenue Contribution (%)",
        text="Revenue_%",
        labels={
            "Revenue_%": "Revenue (%)"
        }
    )

    revenue_chart.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    revenue_chart.update_layout(
        xaxis_title="",
        yaxis_title="Revenue Share (%)"
    )

    st.plotly_chart(
        revenue_chart,
        use_container_width=True
    )


# =========================================================
# 12. KEY BUSINESS INSIGHT
# =========================================================

high_value = segment_summary[
    segment_summary["Segment"]
    == "Highly Active / High Value"
].iloc[0]


st.info(
    f"""
    **💡 Key Business Insight**

    The **Highly Active / High Value** segment contains
    **{high_value['Customer_%']:.2f}%** of customers but
    contributes approximately **{high_value['Revenue_%']:.2f}%**
    of total customer monetary value.

    This indicates that a relatively small group of highly
    engaged customers contributes a large share of the
    business value.
    """
)


st.divider()


# =========================================================
# 13. INTERACTIVE SEGMENT ANALYSIS
# =========================================================

st.subheader("🎯 Explore a Customer Segment")


segment_options = [
    "Highly Active / High Value",
    "Regular / Moderate Value",
    "Inactive / Low Value"
]


selected_segment = st.selectbox(
    "Select a customer segment",
    segment_options
)


selected_data = segment_summary[
    segment_summary["Segment"]
    == selected_segment
].iloc[0]


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Customers",
        f"{selected_data['Customers']:,}"
    )


with metric2:

    st.metric(
        "Avg Recency",
        f"{selected_data['Avg_Recency']:.1f} days"
    )


with metric3:

    st.metric(
        "Avg Frequency",
        f"{selected_data['Avg_Frequency']:.1f}"
    )


with metric4:

    st.metric(
        "Avg Monetary",
        f"₹{selected_data['Avg_Monetary']:,.0f}"
    )


# =========================================================
# 14. BUSINESS RECOMMENDATION
# =========================================================

recommendations = {

    "Highly Active / High Value":
        """
        **Recommended Action: Retention and Loyalty**

        Focus on retaining these valuable customers through
        loyalty programs, premium offers, personalized
        recommendations, and exclusive campaigns.
        """,

    "Regular / Moderate Value":
        """
        **Recommended Action: Increase Engagement**

        Encourage repeat purchases and increase customer
        value through cross-selling, personalized offers,
        and targeted promotions.
        """,

    "Inactive / Low Value":
        """
        **Recommended Action: Re-engagement**

        Use targeted re-engagement campaigns, promotional
        offers, and personalized communication to encourage
        customers to return.
        """
}


st.markdown(
    recommendations[selected_segment]
)


st.divider()


# =========================================================
# 15. RFM PROFILE
# =========================================================

st.subheader("📊 RFM Profile")


rfm_profile = pd.DataFrame({

    "Metric": [
        "Recency",
        "Frequency",
        "Monetary"
    ],

    "Value": [
        selected_data["Avg_Recency"],
        selected_data["Avg_Frequency"],
        selected_data["Avg_Monetary"]
    ]
})


profile_chart = px.bar(
    rfm_profile,
    x="Metric",
    y="Value",
    title=f"Average RFM Values — {selected_segment}",
    text="Value"
)


profile_chart.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)


st.plotly_chart(
    profile_chart,
    use_container_width=True
)


st.divider()


# =========================================================
# 16. PCA VISUALIZATION
# =========================================================

st.subheader("🔬 Customer Segmentation — PCA Visualization")


pca_fig = px.scatter(
    rfm,
    x="PC1",
    y="PC2",
    color="Segment",
    hover_data=[
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary"
    ],
    title="Customer Segments in PCA Space"
)


pca_fig.update_layout(
    xaxis_title="Principal Component 1",
    yaxis_title="Principal Component 2"
)


st.plotly_chart(
    pca_fig,
    use_container_width=True
)


st.caption(
    f"""
    PC1 explains {explained_variance[0]:.2f}% of the variance
    and PC2 explains {explained_variance[1]:.2f}%.
    Together, they explain approximately
    {explained_variance.sum():.2f}% of the variance.

    PCA is used here only for visualization; clustering was
    performed using the standardized RFM features.
    """
)


st.divider()


# =========================================================
# 17. CUSTOMER LOOKUP
# =========================================================

st.subheader("🔎 Customer Lookup")


customer_id = st.number_input(
    "Enter Customer ID",
    min_value=0,
    step=1,
    value=12347
)


customer = rfm[
    rfm["CustomerID"] == customer_id
]


if not customer.empty:

    customer_data = customer.iloc[0]

    st.success(
        f"Customer {int(customer_data['CustomerID'])} "
        f"belongs to **{customer_data['Segment']}**."
    )


    lookup1, lookup2, lookup3 = st.columns(3)


    with lookup1:

        st.metric(
            "Recency",
            f"{customer_data['Recency']:.0f} days"
        )


    with lookup2:

        st.metric(
            "Frequency",
            f"{customer_data['Frequency']:.0f} orders"
        )


    with lookup3:

        st.metric(
            "Monetary",
            f"₹{customer_data['Monetary']:,.2f}"
        )


    st.markdown(
        recommendations[
            customer_data["Segment"]
        ]
    )


else:

    st.warning(
        "Customer ID was not found in the dataset."
    )


# =========================================================
# 18. FOOTER
# =========================================================

st.divider()

st.caption(
    """
    Online Retail Customer Segmentation |
    RFM Analysis + K-Means Clustering |
    Streamlit Dashboard
    """
)