"""
NovaMart Analytics Dashboard
A comprehensive marketing and sales analytics dashboard for omnichannel retail
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Configuration - MUST BE FIRST
st.set_page_config(
    page_title="NovaMart Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A5F; text-align: center; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1rem; color: #666; text-align: center; margin-bottom: 2rem;}
    .insight-box {background-color: #e8f4f8; border-left: 4px solid #17a2b8; padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all datasets"""
    data = {}
    data['campaign'] = pd.read_csv('data/campaign_performance.csv', parse_dates=['date'])
    data['customers'] = pd.read_csv('data/customer_data.csv')
    data['products'] = pd.read_csv('data/product_sales.csv')
    data['leads'] = pd.read_csv('data/lead_scoring_results.csv')
    data['geographic'] = pd.read_csv('data/geographic_data.csv')
    data['funnel'] = pd.read_csv('data/funnel_data.csv')
    data['attribution'] = pd.read_csv('data/channel_attribution.csv')
    data['correlation'] = pd.read_csv('data/correlation_matrix.csv', index_col=0)
    data['journey'] = pd.read_csv('data/customer_journey.csv')
    data['feature_importance'] = pd.read_csv('data/feature_importance.csv')
    data['learning_curve'] = pd.read_csv('data/learning_curve.csv')
    return data


def format_currency(value):
    if value >= 10000000:
        return f"₹{value/10000000:.1f}Cr"
    elif value >= 100000:
        return f"₹{value/100000:.1f}L"
    return f"₹{value:,.0f}"


def format_number(value):
    if value >= 1000000:
        return f"{value/1000000:.1f}M"
    elif value >= 1000:
        return f"{value/1000:.1f}K"
    return f"{value:,.0f}"


# Load data
data = load_data()

# Sidebar
st.sidebar.markdown("## 🛒 NovaMart Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📊 Navigation",
    ["🏠 Executive Summary", 
     "📈 Campaign Performance",
     "👥 Customer Analytics",
     "📦 Product Sales",
     "🎯 Lead Scoring AI",
     "🗺️ Geographic Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Range")
min_date = data['campaign']['date'].min().date()
max_date = data['campaign']['date'].max().date()
date_range = st.sidebar.date_input("Select Period", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Filter data
if len(date_range) == 2:
    campaign_filtered = data['campaign'][
        (data['campaign']['date'].dt.date >= date_range[0]) &
        (data['campaign']['date'].dt.date <= date_range[1])
    ]
else:
    campaign_filtered = data['campaign']


# ==================== PAGE 1: EXECUTIVE SUMMARY ====================
if page == "🏠 Executive Summary":
    st.markdown('<h1 class="main-header">🛒 NovaMart Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Omnichannel Retail Performance | Electronics • Fashion • Home & Living</p>', unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    total_revenue = campaign_filtered['revenue'].sum()
    total_spend = campaign_filtered['spend'].sum()
    total_conversions = campaign_filtered['conversions'].sum()
    avg_roas = total_revenue / total_spend if total_spend > 0 else 0
    total_customers = len(data['customers'])
    
    col1.metric("💰 Total Revenue", format_currency(total_revenue), "+12.5%")
    col2.metric("📊 Ad Spend", format_currency(total_spend), "-3.2%")
    col3.metric("🎯 Conversions", format_number(total_conversions), "+8.7%")
    col4.metric("📈 Avg ROAS", f"{avg_roas:.2f}x", "+0.3x")
    col5.metric("👥 Customers", format_number(total_customers), "+15.2%")
    
    st.markdown("---")
    
    # Row 1
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Revenue & Spend Trend")
        daily = campaign_filtered.groupby('date').agg({'revenue': 'sum', 'spend': 'sum'}).reset_index()
        daily['revenue_ma'] = daily['revenue'].rolling(7, min_periods=1).mean()
        daily['spend_ma'] = daily['spend'].rolling(7, min_periods=1).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['revenue_ma'], name='Revenue (7-day avg)', 
                                  line=dict(color='#2ecc71', width=2), fill='tozeroy', fillcolor='rgba(46,204,113,0.1)'))
        fig.add_trace(go.Scatter(x=daily['date'], y=daily['spend_ma'], name='Spend (7-day avg)', 
                                  line=dict(color='#e74c3c', width=2)))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation='h', y=1.02), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Channel Revenue")
        channel_rev = campaign_filtered.groupby('channel')['revenue'].sum().reset_index()
        fig = px.pie(channel_rev, values='revenue', names='channel', hole=0.4, 
                     color_discrete_sequence=['#3498db','#2ecc71','#f39c12','#e74c3c','#9b59b6'])
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Conversion Funnel")
        funnel = data['funnel']
        fig = go.Figure(go.Funnel(
            y=funnel['stage'], x=funnel['visitors'],
            textposition="inside", textinfo="value+percent initial",
            marker=dict(color=['#3498db','#2ecc71','#f39c12','#e67e22','#e74c3c','#9b59b6'])
        ))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Channel Attribution")
        attr = data['attribution'].melt(id_vars=['channel'], var_name='Model', value_name='Attribution %')
        fig = px.bar(attr, x='channel', y='Attribution %', color='Model', barmode='group',
                     color_discrete_sequence=['#3498db','#2ecc71','#f39c12','#e74c3c','#9b59b6'])
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation='h', y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Journey
    st.subheader("🛤️ Customer Journey Paths")
    journey = data['journey'].copy()
    all_nodes = []
    for col in ['touchpoint_1','touchpoint_2','touchpoint_3','touchpoint_4']:
        all_nodes.extend(journey[col].dropna().unique())
    all_nodes = list(dict.fromkeys(all_nodes))
    node_idx = {n:i for i,n in enumerate(all_nodes)}
    
    sources, targets, values = [], [], []
    for _, row in journey.iterrows():
        tps = [row['touchpoint_1'], row['touchpoint_2'], row['touchpoint_3'], row['touchpoint_4']]
        tps = [t for t in tps if pd.notna(t)]
        for i in range(len(tps)-1):
            sources.append(node_idx[tps[i]])
            targets.append(node_idx[tps[i+1]])
            values.append(row['customer_count'])
    
    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, label=all_nodes, color=['#3498db','#2ecc71','#f39c12','#e74c3c','#9b59b6','#1abc9c','#e67e22','#95a5a6'][:len(all_nodes)]),
        link=dict(source=sources, target=targets, value=values, color='rgba(52,152,219,0.3)')
    ))
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE 2: CAMPAIGN PERFORMANCE ====================
elif page == "📈 Campaign Performance":
    st.markdown("## 📈 Campaign Performance Analytics")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    channels = ['All'] + list(campaign_filtered['channel'].unique())
    regions = ['All'] + list(campaign_filtered['region'].unique())
    types = ['All'] + list(campaign_filtered['campaign_type'].unique())
    
    sel_channel = col1.selectbox("🔗 Channel", channels)
    sel_region = col2.selectbox("📍 Region", regions)
    sel_type = col3.selectbox("📋 Type", types)
    
    df = campaign_filtered.copy()
    if sel_channel != 'All': df = df[df['channel'] == sel_channel]
    if sel_region != 'All': df = df[df['region'] == sel_region]
    if sel_type != 'All': df = df[df['campaign_type'] == sel_type]
    
    # KPIs
    tot_imp = df['impressions'].sum()
    tot_clicks = df['clicks'].sum()
    tot_conv = df['conversions'].sum()
    avg_ctr = (tot_clicks/tot_imp*100) if tot_imp > 0 else 0
    avg_cvr = (tot_conv/tot_clicks*100) if tot_clicks > 0 else 0
    avg_cpa = df['spend'].sum()/tot_conv if tot_conv > 0 else 0
    
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("👁️ Impressions", format_number(tot_imp))
    c2.metric("🖱️ Clicks", format_number(tot_clicks))
    c3.metric("🎯 Conversions", format_number(tot_conv))
    c4.metric("📈 CTR", f"{avg_ctr:.2f}%")
    c5.metric("🔄 Conv Rate", f"{avg_cvr:.2f}%")
    c6.metric("💵 Avg CPA", format_currency(avg_cpa))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Channel Performance")
        ch_perf = df.groupby('channel').agg({'spend':'sum','revenue':'sum','conversions':'sum'}).reset_index()
        ch_perf['roas'] = ch_perf['revenue']/ch_perf['spend']
        ch_perf = ch_perf.sort_values('revenue', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(y=ch_perf['channel'], x=ch_perf['revenue'], name='Revenue', orientation='h', marker_color='#2ecc71'))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Day-of-Week Performance")
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = df.groupby('day_of_week').agg({'conversions':'sum','revenue':'sum'}).reindex(dow_order).reset_index()
        
        fig = go.Figure(go.Bar(x=dow['day_of_week'], y=dow['conversions'], marker_color='#3498db'))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("🗺️ Regional Channel ROAS Heatmap")
    rc = df.groupby(['region','channel']).agg({'roas':'mean'}).reset_index()
    pivot = rc.pivot(index='region', columns='channel', values='roas')
    
    fig = px.imshow(pivot, color_continuous_scale='RdYlGn', aspect='auto')
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation
    st.subheader("🔗 Marketing Metrics Correlation")
    fig = px.imshow(data['correlation'], color_continuous_scale='RdBu_r', aspect='auto', zmin=-1, zmax=1)
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE 3: CUSTOMER ANALYTICS ====================
elif page == "👥 Customer Analytics":
    st.markdown("## 👥 Customer Analytics & Segmentation")
    st.markdown("---")
    
    cust = data['customers']
    
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💎 Total LTV", format_currency(cust['lifetime_value'].sum()))
    c2.metric("📊 Avg LTV", format_currency(cust['lifetime_value'].mean()))
    c3.metric("⚠️ Churn Rate", f"{cust['is_churned'].mean()*100:.1f}%")
    c4.metric("⭐ Avg Satisfaction", f"{cust['satisfaction_score'].mean():.2f}/5")
    c5.metric("👑 Premium %", f"{(cust['customer_segment']=='Premium').mean()*100:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Customer Segments")
        seg = cust['customer_segment'].value_counts().reset_index()
        seg.columns = ['Segment','Count']
        colors = {'Premium':'#f39c12','Regular':'#3498db','Budget':'#2ecc71','New':'#9b59b6','Churned':'#e74c3c'}
        fig = px.pie(seg, values='Count', names='Segment', hole=0.4, color='Segment', color_discrete_map=colors)
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 LTV by Segment")
        ltv = cust.groupby('customer_segment')['lifetime_value'].mean().reset_index()
        ltv = ltv.sort_values('lifetime_value', ascending=True)
        fig = go.Figure(go.Bar(y=ltv['customer_segment'], x=ltv['lifetime_value'], orientation='h',
                               marker_color=['#e74c3c','#2ecc71','#9b59b6','#3498db','#f39c12']))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Age Distribution")
        colors = {'Premium':'#f39c12','Regular':'#3498db','Budget':'#2ecc71','New':'#9b59b6','Churned':'#e74c3c'}
        fig = px.histogram(cust, x='age', color='customer_segment', nbins=30, barmode='stack',
                          color_discrete_map=colors)
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation='h', y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📢 Acquisition Channel LTV")
        ch = cust.groupby('acquisition_channel').agg({'lifetime_value':'mean','is_churned':'mean'}).reset_index()
        ch = ch.sort_values('lifetime_value', ascending=False)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=ch['acquisition_channel'], y=ch['lifetime_value'], name='Avg LTV', marker_color='#3498db'), secondary_y=False)
        fig.add_trace(go.Scatter(x=ch['acquisition_channel'], y=ch['is_churned']*100, name='Churn %', mode='lines+markers', line=dict(color='#e74c3c', width=3)), secondary_y=True)
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation='h', y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn Analysis
    st.subheader("⚠️ Churn Risk Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        churn_seg = cust.groupby('customer_segment')['is_churned'].mean().reset_index()
        churn_seg['is_churned'] = churn_seg['is_churned'] * 100
        churn_seg = churn_seg.sort_values('is_churned', ascending=True)
        
        fig = go.Figure(go.Bar(y=churn_seg['customer_segment'], x=churn_seg['is_churned'], orientation='h',
                               marker_color=['#2ecc71','#3498db','#f39c12','#e67e22','#e74c3c']))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), title="Churn Rate by Segment")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        active = cust[cust['is_churned']==0]
        fig = px.histogram(active, x='churn_probability', nbins=50, color_discrete_sequence=['#3498db'])
        fig.add_vline(x=0.5, line_dash="dash", line_color="red")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), title="Churn Probability (Active Customers)")
        st.plotly_chart(fig, use_container_width=True)
    
    high_risk = cust[(cust['is_churned']==0) & (cust['churn_probability']>0.25)]
    st.markdown(f"""
    <div class="insight-box">
        <strong>⚠️ At-Risk:</strong> {len(high_risk):,} customers with >25% churn probability | 
        <strong>💰 Revenue at Risk:</strong> {format_currency(high_risk['lifetime_value'].sum())}
    </div>
    """, unsafe_allow_html=True)


# ==================== PAGE 4: PRODUCT SALES ====================
elif page == "📦 Product Sales":
    st.markdown("## 📦 Product Sales Analytics")
    st.markdown("---")
    
    prod = data['products']
    
    col1, col2, col3 = st.columns(3)
    cats = ['All'] + list(prod['category'].unique())
    regs = ['All'] + list(prod['region'].unique())
    years = ['All'] + sorted(prod['year'].unique())
    
    sel_cat = col1.selectbox("📂 Category", cats)
    sel_reg = col2.selectbox("📍 Region", regs)
    sel_yr = col3.selectbox("📅 Year", years)
    
    df = prod.copy()
    if sel_cat != 'All': df = df[df['category'] == sel_cat]
    if sel_reg != 'All': df = df[df['region'] == sel_reg]
    if sel_yr != 'All': df = df[df['year'] == sel_yr]
    
    # KPIs
    tot_sales = df['sales'].sum()
    tot_profit = df['profit'].sum()
    avg_margin = (tot_profit/tot_sales*100) if tot_sales > 0 else 0
    tot_units = df['units_sold'].sum()
    avg_rating = df['avg_rating'].mean()
    
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Total Sales", format_currency(tot_sales))
    c2.metric("📈 Total Profit", format_currency(tot_profit))
    c3.metric("📊 Profit Margin", f"{avg_margin:.1f}%")
    c4.metric("📦 Units Sold", format_number(tot_units))
    c5.metric("⭐ Avg Rating", f"{avg_rating:.1f}/5")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Category Performance")
        cat_perf = df.groupby('category').agg({'sales':'sum','profit':'sum'}).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cat_perf['category'], y=cat_perf['sales'], name='Sales', marker_color='#3498db'))
        fig.add_trace(go.Bar(x=cat_perf['category'], y=cat_perf['profit'], name='Profit', marker_color='#2ecc71'))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), barmode='group', legend=dict(orientation='h', y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Quarterly Sales Trend")
        qtr = df.groupby(['year','quarter']).agg({'sales':'sum'}).reset_index()
        qtr['period'] = qtr['quarter'] + ' ' + qtr['year'].astype(str)
        
        fig = px.line(qtr, x='period', y='sales', markers=True)
        fig.update_traces(line_color='#3498db', line_width=3)
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Products")
        top = df.groupby('product_name')['sales'].sum().reset_index().sort_values('sales', ascending=True).tail(10)
        
        fig = go.Figure(go.Bar(y=top['product_name'], x=top['sales'], orientation='h', marker_color='#3498db'))
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Profit Margin by Subcategory")
        sub = df.groupby('subcategory').agg({'sales':'sum','profit':'sum'}).reset_index()
        sub['margin'] = (sub['profit']/sub['sales'])*100
        sub = sub.sort_values('margin', ascending=True)
        
        colors = ['#e74c3c' if m<15 else '#f39c12' if m<20 else '#2ecc71' for m in sub['margin']]
        fig = go.Figure(go.Bar(y=sub['subcategory'], x=sub['margin'], orientation='h', marker_color=colors))
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE 5: LEAD SCORING AI ====================
elif page == "🎯 Lead Scoring AI":
    st.markdown("## 🎯 Lead Scoring AI Model Performance")
    st.markdown("---")
    
    leads = data['leads']
    fi = data['feature_importance']
    lc = data['learning_curve']
    
    # Metrics
    actual_pos = leads['actual_converted'].sum()
    pred_pos = leads['predicted_class'].sum()
    tp = ((leads['actual_converted']==1) & (leads['predicted_class']==1)).sum()
    tn = ((leads['actual_converted']==0) & (leads['predicted_class']==0)).sum()
    fp = ((leads['actual_converted']==0) & (leads['predicted_class']==1)).sum()
    fn = ((leads['actual_converted']==1) & (leads['predicted_class']==0)).sum()
    
    precision = tp/pred_pos if pred_pos > 0 else 0
    recall = tp/actual_pos if actual_pos > 0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    accuracy = (tp+tn)/len(leads)
    
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🎯 Accuracy", f"{accuracy:.1%}")
    c2.metric("📈 Precision", f"{precision:.1%}")
    c3.metric("🔄 Recall", f"{recall:.1%}")
    c4.metric("⚖️ F1 Score", f"{f1:.1%}")
    c5.metric("📊 Total Leads", format_number(len(leads)))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confusion Matrix")
        conf = [[tn, fp], [fn, tp]]
        fig = go.Figure(go.Heatmap(
            z=conf, x=['Pred: No','Pred: Yes'], y=['Actual: No','Actual: Yes'],
            colorscale='Blues', text=conf, texttemplate="%{text}", textfont={"size":20}
        ))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Feature Importance")
        fi_sorted = fi.sort_values('importance', ascending=True)
        fig = go.Figure(go.Bar(y=fi_sorted['feature'], x=fi_sorted['importance'], orientation='h', marker_color='#3498db'))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Learning Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lc['training_size'], y=lc['train_score'], name='Train', line=dict(color='#3498db', width=2), mode='lines+markers'))
        fig.add_trace(go.Scatter(x=lc['training_size'], y=lc['validation_score'], name='Validation', line=dict(color='#2ecc71', width=2), mode='lines+markers'))
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation='h', y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Probability Distribution")
        converted = leads[leads['actual_converted']==1]['predicted_probability']
        not_converted = leads[leads['actual_converted']==0]['predicted_probability']
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=not_converted, name='Not Converted', marker_color='#e74c3c', opacity=0.7, nbinsx=30))
        fig.add_trace(go.Histogram(x=converted, name='Converted', marker_color='#2ecc71', opacity=0.7, nbinsx=30))
        fig.add_vline(x=0.5, line_dash="dash", line_color="black")
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), barmode='overlay', legend=dict(orientation='h', y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    
    # By company size
    st.subheader("📈 Conversion by Company Size")
    size_order = ['Small','Medium','Large','Enterprise']
    size_conv = leads.groupby('company_size').agg({'actual_converted':['sum','count','mean']}).reset_index()
    size_conv.columns = ['Size','Conversions','Total','Rate']
    size_conv['Size'] = pd.Categorical(size_conv['Size'], categories=size_order, ordered=True)
    size_conv = size_conv.sort_values('Size')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=size_conv['Size'], y=size_conv['Total'], name='Total Leads', marker_color='#3498db'), secondary_y=False)
    fig.add_trace(go.Scatter(x=size_conv['Size'], y=size_conv['Rate']*100, name='Conv Rate %', mode='lines+markers', line=dict(color='#e74c3c', width=3)), secondary_y=True)
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation='h', y=1.02))
    st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE 6: GEOGRAPHIC INSIGHTS ====================
elif page == "🗺️ Geographic Insights":
    st.markdown("## 🗺️ Geographic Performance Insights")
    st.markdown("---")
    
    geo = data['geographic']
    
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Total Revenue", format_currency(geo['total_revenue'].sum()))
    c2.metric("👥 Total Customers", format_number(geo['total_customers'].sum()))
    c3.metric("🏪 Total Stores", str(geo['store_count'].sum()))
    c4.metric("📊 Avg Penetration", f"{geo['market_penetration'].mean():.1f}%")
    c5.metric("📈 Avg YoY Growth", f"{geo['yoy_growth'].mean():.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Revenue by State")
        fig = px.scatter_geo(geo, lat='latitude', lon='longitude', size='total_revenue', color='yoy_growth',
                            hover_name='state', color_continuous_scale='RdYlGn', scope='asia')
        fig.update_geos(center=dict(lat=22, lon=78), projection_scale=4, showcountries=True)
        fig.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Top 5 States")
        top5 = geo.nlargest(5, 'total_revenue')[['state','total_revenue','yoy_growth']]
        for _, r in top5.iterrows():
            icon = '🟢' if r['yoy_growth'] > 10 else '🟡' if r['yoy_growth'] > 0 else '🔴'
            st.markdown(f"**{r['state']}**\n{format_currency(r['total_revenue'])} | {icon} {r['yoy_growth']:.1f}%")
            st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Regional Distribution")
        reg = geo.groupby('region')['total_revenue'].sum().reset_index()
        fig = px.pie(reg, values='total_revenue', names='region', hole=0.4,
                     color_discrete_sequence=['#3498db','#2ecc71','#f39c12','#e74c3c','#9b59b6'])
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 YoY Growth by State")
        geo_sorted = geo.sort_values('yoy_growth', ascending=True)
        colors = ['#e74c3c' if g<0 else '#f39c12' if g<10 else '#2ecc71' for g in geo_sorted['yoy_growth']]
        
        fig = go.Figure(go.Bar(y=geo_sorted['state'], x=geo_sorted['yoy_growth'], orientation='h', marker_color=colors))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 State Metrics")
    display = geo[['state','region','total_customers','total_revenue','revenue_per_customer','store_count','market_penetration','yoy_growth','customer_satisfaction']].copy()
    display.columns = ['State','Region','Customers','Revenue','Rev/Cust','Stores','Penetration %','YoY %','Satisfaction']
    st.dataframe(display, use_container_width=True, height=400)


# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#666; padding:1rem;'>🛒 NovaMart Analytics Dashboard | Built with Streamlit & Plotly</div>", unsafe_allow_html=True)
