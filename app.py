# ===============================
# NovaMart Executive Dashboard
# Single-file Streamlit App
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="NovaMart Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NovaMart Omnichannel Analytics Dashboard")
st.caption("Simple, clear insights for leadership decisions")

# -------------------------------
# DATA LOADER
# -------------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# -------------------------------
# LOAD DATA
# -------------------------------
campaign = load_csv("data/campaign_performance.csv")
customers = load_csv("data/customer_data.csv")
products = load_csv("data/product_sales.csv")
geo = load_csv("data/geographic_data.csv")
attrib = load_csv("data/channel_attribution.csv")
funnel = load_csv("data/funnel_data.csv")
corr = load_csv("data/correlation_matrix.csv")
leads = load_csv("data/lead_scoring_results.csv")
learning = load_csv("data/learning_curve.csv")
features = load_csv("data/feature_importance.csv")

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Campaign Analytics",
        "Customer Insights",
        "Product Performance",
        "Geographic Analysis",
        "Attribution & Funnel",
        "ML Model Evaluation"
    ]
)

# =====================================================
# 1️⃣ EXECUTIVE OVERVIEW
# =====================================================
if page == "Executive Overview":

    st.header("📌 Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Revenue", f"₹{campaign.revenue.sum():,.0f}")
    col2.metric("Conversions", f"{campaign.conversions.sum():,.0f}")
    col3.metric("ROAS", f"{campaign.revenue.sum() / campaign.spend.sum():.2f}")
    col4.metric("Customers", customers.customer_id.nunique())

    st.divider()

    metric = st.selectbox("Channel Metric", ["revenue", "conversions", "roas"])
    channel_chart = px.bar(
        campaign.groupby("channel")[metric].sum().reset_index(),
        x=metric,
        y="channel",
        orientation="h",
        title="Channel Performance"
    )
    st.plotly_chart(channel_chart, use_container_width=True)

    campaign["date"] = pd.to_datetime(campaign["date"])
    trend = campaign.groupby("date")["revenue"].sum().reset_index()
    st.plotly_chart(
        px.line(trend, x="date", y="revenue", title="Revenue Trend"),
        use_container_width=True
    )

# =====================================================
# 2️⃣ CAMPAIGN ANALYTICS
# =====================================================
elif page == "Campaign Analytics":

    st.header("📈 Campaign Analytics")

    campaign["month"] = pd.to_datetime(campaign["date"]).dt.to_period("M").astype(str)

    stacked = px.bar(
        campaign,
        x="month",
        y="spend",
        color="campaign_type",
        title="Monthly Spend by Campaign Type"
    )
    st.plotly_chart(stacked, use_container_width=True)

    heat = campaign.pivot_table(
        index=campaign["date"].astype(str),
        values="revenue",
        aggfunc="sum"
    )
    st.plotly_chart(
        px.imshow(heat.T, title="Daily Revenue Heatmap"),
        use_container_width=True
    )

# =====================================================
# 3️⃣ CUSTOMER INSIGHTS
# =====================================================
elif page == "Customer Insights":

    st.header("👥 Customer Insights")

    bins = st.slider("Age Bin Size", 5, 20, 10)

    st.plotly_chart(
        px.histogram(customers, x="age", nbins=bins, title="Age Distribution"),
        use_container_width=True
    )

    st.plotly_chart(
        px.box(
            customers,
            x="customer_segment",
            y="lifetime_value",
            title="Lifetime Value by Segment"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(
            customers,
            x="income",
            y="lifetime_value",
            color="customer_segment",
            title="Income vs Lifetime Value"
        ),
        use_container_width=True
    )

# =====================================================
# 4️⃣ PRODUCT PERFORMANCE
# =====================================================
elif page == "Product Performance":

    st.header("📦 Product Performance")

    treemap = px.treemap(
        products,
        path=["category", "subcategory", "product"],
        values="sales",
        color="profit_margin",
        title="Product Sales & Margin Overview"
    )
    st.plotly_chart(treemap, use_container_width=True)

# =====================================================
# 5️⃣ GEOGRAPHIC ANALYSIS
# =====================================================
elif page == "Geographic Analysis":

    st.header("🗺️ Geographic Performance")

    metric = st.selectbox(
        "Metric",
        ["revenue", "customers", "market_penetration", "yoy_growth"]
    )

    st.plotly_chart(
        px.choropleth(
            geo,
            geojson="https://gist.githubusercontent.com/jbrooksuk/0a1c2f4b3f5c6c7c8/raw/india_states.geojson",
            locations="state",
            featureidkey="properties.ST_NM",
            color=metric,
            title="State-wise Performance"
        ),
        use_container_width=True
    )

# =====================================================
# 6️⃣ ATTRIBUTION & FUNNEL
# =====================================================
elif page == "Attribution & Funnel":

    st.header("🔄 Attribution & Funnel")

    model = st.selectbox("Attribution Model", ["first_touch", "last_touch", "linear"])

    donut = px.pie(
        attrib,
        names="channel",
        values=model,
        hole=0.5,
        title="Attribution Model Comparison"
    )
    st.plotly_chart(donut, use_container_width=True)

    funnel_fig = go.Figure(go.Funnel(
        y=funnel.stage,
        x=funnel.count
    ))
    funnel_fig.update_layout(title="Marketing Conversion Funnel")
    st.plotly_chart(funnel_fig, use_container_width=True)

    st.plotly_chart(
        px.imshow(corr.set_index("metric"), title="Correlation Heatmap"),
        use_container_width=True
    )

# =====================================================
# 7️⃣ ML MODEL EVALUATION
# =====================================================
elif page == "ML Model Evaluation":

    st.header("🤖 ML Model Evaluation")

    threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.5)

    leads["predicted"] = (leads["predicted_probability"] >= threshold).astype(int)
    cm = confusion_matrix(leads.actual_converted, leads.predicted)

    st.plotly_chart(
        px.imshow(cm, text_auto=True, title="Confusion Matrix"),
        use_container_width=True
    )

    fpr, tpr, _ = roc_curve(leads.actual_converted, leads.predicted_probability)
    roc_auc = auc(fpr, tpr)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.2f}"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random"))
    roc_fig.update_layout(title="ROC Curve")
    st.plotly_chart(roc_fig, use_container_width=True)

    lc = px.line(
        learning,
        x="training_size",
        y=["train_score", "validation_score"],
        title="Learning Curve"
    )
    st.plotly_chart(lc, use_container_width=True)

    fi = px.bar(
        features.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance"
    )
    st.plotly_chart(fi, use_container_width=True)
