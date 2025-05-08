"""
Streamlit App: Mall Customers Clustering & Association Analysis Dashboard
This app performs clustering (KMeans, Hierarchical) and association rule mining
on the Mall_Customers dataset with an elegant and interactive UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title='Mall Customers Analysis',
    page_icon=':bar_chart:',
    layout='wide'
)

# --- Sidebar: Data Upload ---
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Mall_Customers.csv", type=["csv"])
def load_data(uploaded_file=None):
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None
    else:
        if os.path.exists("Mall_Customers.csv"):
            try:
                data = pd.read_csv("Mall_Customers.csv")
            except Exception as e:
                st.error(f"Error reading local Mall_Customers.csv: {e}")
                return None
        else:
            st.error("Mall_Customers.csv not found. Please upload a dataset.")
            return None
    return data

df = load_data(uploaded_file)
if df is None:
    st.stop()

# Validate required columns
required_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
if not all(col in df.columns for col in required_cols):
    st.error(f"Dataset is missing required columns: {required_cols}")
    st.stop()

# --- Main Title and Description ---
st.title("Mall Customers Data Analysis Dashboard")
st.markdown(
    """
    Explore customer data with interactive clustering and association rule mining.
    Use the tabs below to navigate between data preview, clustering analysis, and association rules.
    """
)

# --- Define Tabs ---
tab1, tab2, tab3 = st.tabs(["Data Preview", "Clustering", "Association Rules"])

# --- Tab 1: Data Preview ---
with tab1:
    st.header("Data Preview")
    st.markdown("Toggle between raw data and preprocessed (scaled & binned) data.")
    view_option = st.radio("View:", ("Raw Data", "Preprocessed Data"))
    
    if view_option == "Raw Data":
        st.write(df)
    else:
        # Preprocess data for preview: scale and add binned categories
        df_pre = df.copy()
        scaler = StandardScaler()
        df_pre[required_cols] = scaler.fit_transform(df_pre[required_cols])
        # Quantile binning into 3 categories
        df_pre['Age_Bin'] = pd.qcut(df['Age'], 3, labels=['Young','Adult','Senior'])
        df_pre['Income_Bin'] = pd.qcut(df['Annual Income (k$)'], 3, labels=['Low','Medium','High'])
        df_pre['Score_Bin'] = pd.qcut(df['Spending Score (1-100)'], 3, labels=['Low','Medium','High'])
        st.write(df_pre)

# --- Tab 2: Clustering ---
with tab2:
    st.header("Clustering Analysis")
    # Prepare scaled data for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[required_cols])
    
    # --- K-Means Clustering ---
    st.subheader("K-Means Clustering")
    st.markdown("Use K-Means to cluster customers by Age, Income, and Score. Adjust *k* and explore the clusters.")
    # Elbow plot to help choose k
    sse = []
    k_values = range(1, 11)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        sse.append(km.inertia_)
    fig_elbow = px.line(x=list(k_values), y=sse, markers=True,
                        labels={'x': 'Number of clusters (k)', 'y': 'SSE'},
                        title="Elbow Method: SSE vs k")
    st.plotly_chart(fig_elbow, use_container_width=True)

    # K selection
    k = st.slider("Select number of clusters (K)", 2, 10, 3)
    km = KMeans(n_clusters=k, random_state=42)
    labels_km = km.fit_predict(X_scaled)
    df_km = df.copy()
    df_km['KMeans_Cluster'] = labels_km.astype(str)

    # 3D scatter of K-Means clusters
    fig_km = px.scatter_3d(
        df_km, 
        x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
        color='KMeans_Cluster',
        title=f"3D Scatter (KMeans, k={k})",
        width=800, height=600
    )
    st.plotly_chart(fig_km, use_container_width=True)

    # Download KMeans results
    csv_km = df_km.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download K-Means Clusters Data",
        csv_km,
        file_name="kmeans_clusters.csv",
        mime="text/csv"
    )

    # --- Hierarchical Clustering ---
    st.subheader("Hierarchical Clustering")
    st.markdown("Use Agglomerative clustering. Adjust the number of clusters to cut the dendrogram.")
    # Compute linkage matrix
    Z = linkage(X_scaled, method='ward')
    # Plot dendrogram (truncated for readability)
    fig_dend = plt.figure(figsize=(8, 5))
    dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=10.)
    plt.title("Hierarchical Clustering Dendrogram (truncated)")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    st.pyplot(fig_dend)
    plt.close(fig_dend)

    # Hierarchical cluster selection
    hc_clusters = st.slider("Number of clusters", 2, 10, 3)
    labels_hc = fcluster(Z, t=hc_clusters, criterion='maxclust')
    df_hc = df.copy()
    df_hc['HierCluster'] = labels_hc.astype(str)

    # 3D scatter of Hierarchical clusters
    fig_hc = px.scatter_3d(
        df_hc,
        x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
        color='HierCluster',
        title=f"3D Scatter (Hierarchical, clusters={hc_clusters})",
        width=800, height=600
    )
    st.plotly_chart(fig_hc, use_container_width=True)

    # Download Hierarchical results
    csv_hc = df_hc.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Hierarchical Clusters Data",
        csv_hc,
        file_name="hierarchical_clusters.csv",
        mime="text/csv"
    )

# --- Tab 3: Association Rules ---
with tab3:
    st.header("Association Rule Mining")
    st.markdown("Explore association rules in binned Age, Income, and Score categories.")

    # Quantile-based binning into 3 categories
    df_assoc = df.copy()
    df_assoc['Age_Bin'] = pd.qcut(df_assoc['Age'], 3, labels=['Young','Adult','Senior'])
    df_assoc['Income_Bin'] = pd.qcut(df_assoc['Annual Income (k$)'], 3, labels=['Low','Medium','High'])
    df_assoc['Score_Bin'] = pd.qcut(df_assoc['Spending Score (1-100)'], 3, labels=['Low','Medium','High'])

    # One-hot encode the binned categories
    df_onehot = pd.get_dummies(df_assoc[['Age_Bin','Income_Bin','Score_Bin']])

    # Sliders for support and confidence
    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.1, step=0.01)
    min_confidence = st.slider("Minimum Confidence", 0.01, 1.0, 0.5, step=0.01)

    # Generate frequent itemsets and association rules
    try:
        freq_items = apriori(df_onehot, min_support=min_support, use_colnames=True)
        if freq_items.empty:
            st.warning("No frequent itemsets found with the given support. Try lowering the support threshold.")
        else:
            rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
            if rules.empty:
                st.warning("No association rules found with the given thresholds. Adjust support/confidence sliders.")
            else:
                # Prepare rules for display
                rules_display = rules[['antecedents','consequents','support','confidence','lift']].copy()
                # Convert frozensets to readable strings
                rules_display['Antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules_display['Consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                rules_display.drop(['antecedents','consequents'], axis=1, inplace=True)
                rules_display.rename(columns={'support':'Support','confidence':'Confidence','lift':'Lift'}, inplace=True)
                # Clean prefix text for readability
                rules_display['Antecedents'] = rules_display['Antecedents'] \
                    .str.replace("Age_Bin_", "Age: ", regex=False) \
                    .str.replace("Income_Bin_", "Income: ", regex=False) \
                    .str.replace("Score_Bin_", "Score: ", regex=False)
                rules_display['Consequents'] = rules_display['Consequents'] \
                    .str.replace("Age_Bin_", "Age: ", regex=False) \
                    .str.replace("Income_Bin_", "Income: ", regex=False) \
                    .str.replace("Score_Bin_", "Score: ", regex=False)
                # Sorting option
                sort_col = st.selectbox("Sort rules by:", options=['Lift','Confidence','Support'])
                rules_display.sort_values(by=sort_col, ascending=False, inplace=True)
                # Display rules
                st.dataframe(rules_display, use_container_width=True)
                # Download rules
                csv_rules = rules_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Association Rules CSV",
                    csv_rules,
                    file_name="association_rules.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"Error generating association rules: {e}")
