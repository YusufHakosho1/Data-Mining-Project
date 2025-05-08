import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
import plotly.express as px
import plotly.figure_factory as ff
from mlxtend.frequent_patterns import apriori, association_rules

# Set page config
st.set_page_config(page_title="Mall Customer Data Analysis", layout="wide")
# Header title
st.title("🛍️ Mall Customer Data Analysis Dashboard")

# Sidebar - file upload and controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Display options
show_raw = st.sidebar.checkbox("Show Raw Data", value=True)
show_pre = st.sidebar.checkbox("Show Preprocessed Data", value=True)

# Clustering parameters (to be set after data loading)
k = None
n_clusters_h = None

# Load and validate dataset
needed_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df = None

# Try loading uploaded file
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        df = None

# Check for required columns in uploaded file
if df is not None:
    if not set(needed_cols).issubset(df.columns):
        st.error("Uploaded data is missing required columns. Using default dataset instead.")
        df = None

# If no valid upload, try loading default Mall_Customers.csv
if df is None:
    try:
        # Attempt to read local file
        df = pd.read_csv("Mall_Customers.csv")
        st.info("Loaded default Mall_Customers.csv from local directory.")
    except:
        try:
            # Attempt to read from GitHub gist URL
            df = pd.read_csv("https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/8bd6144a87988213693754baaa13fb204933282d/Mall_Customers.csv")
            st.info("Loaded default Mall_Customers dataset from remote source.")
        except:
            # If all fails, create synthetic fallback dataset
            np.random.seed(42)
            df = pd.DataFrame({
                'CustomerID': range(1,101),
                'Gender': np.random.choice(['Male','Female'], 100),
                'Age': np.random.randint(18, 70, 100),
                'Annual Income (k$)': np.random.randint(15, 130, 100),
                'Spending Score (1-100)': np.random.randint(1, 101, 100)
            })
            st.warning("Using synthetic default dataset.")
# Final check for required columns
if not set(needed_cols).issubset(df.columns):
    st.error("Default dataset is missing required columns. Cannot proceed.")
    st.stop()

# Prepare preprocessed data (normalize numeric features)
df_preprocessed = df.copy()
scaler = StandardScaler()
df_preprocessed[needed_cols] = scaler.fit_transform(df_preprocessed[needed_cols])

# Determine max k for sliders based on data size
max_k = min(10, len(df))
if max_k < 1:
    st.error("Not enough data for clustering.")
else:
    k = st.sidebar.slider("K for K-Means", min_value=1, max_value=max_k, value=min(3, max_k), step=1)
    n_clusters_h = st.sidebar.slider("Hierarchical Clusters", min_value=1, max_value=max_k, value=min(2, max_k), step=1)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Preview", "Clustering", "Association Rules"])

with tab1:
    st.header("📊 Data Preview")
    if show_raw:
        st.subheader("Raw Data")
        st.dataframe(df)
    if show_pre:
        st.subheader("Preprocessed Data (Normalized Numeric Features)")
        st.dataframe(df_preprocessed)
    if not show_raw and not show_pre:
        st.info("Use the checkboxes in the sidebar to display raw or preprocessed data.")

with tab2:
    st.header("🔍 Clustering")
    # Elbow Method for KMeans
    st.subheader("Elbow Method (SSE vs. k)")
    sse = []
    features = df_preprocessed[needed_cols].values
    for i in range(1, max_k+1):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(features)
        sse.append(km.inertia_)
    fig_elbow = px.line(x=list(range(1, max_k+1)), y=sse, markers=True,
                        labels={'x': 'Number of Clusters k', 'y': 'SSE (Inertia)'},
                        title='Elbow Method for K-Means')
    fig_elbow.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig_elbow, use_container_width=True)

    # K-Means Clustering with chosen k
    st.subheader(f"K-Means Clustering (k = {k})")
    km = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = km.fit_predict(features)
    df_kmeans = df.copy()
    df_kmeans['KMeans Cluster'] = kmeans_labels
    symbol_col = 'Gender' if 'Gender' in df_kmeans.columns else None
    fig_cluster = px.scatter_3d(df_kmeans, 
                                x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                                color='KMeans Cluster', symbol=symbol_col,
                                title='Customer Segments by K-Means')
    st.plotly_chart(fig_cluster, use_container_width=True)

    # Hierarchical Clustering (Single Linkage)
    st.subheader("Hierarchical Clustering (Single Linkage)")
    Z = linkage(features, method='single')
    # Dendrogram
    try:
        fig_dendro = ff.create_dendrogram(features, orientation='bottom')
        fig_dendro.update_layout(width=800, height=400, title='Dendrogram (Single Linkage)')
        st.plotly_chart(fig_dendro, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating dendrogram: {e}")
    # Assign clusters based on selected number
    hier_labels = fcluster(Z, t=n_clusters_h, criterion='maxclust')
    df_kmeans['Hierarchical Cluster'] = hier_labels
    fig_hier = px.scatter_3d(df_kmeans, 
                             x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                             color='Hierarchical Cluster', symbol=symbol_col,
                             title='Customer Segments by Hierarchical Clustering')
    st.plotly_chart(fig_hier, use_container_width=True)

    # Download clustered data
    csv_data = df_kmeans.to_csv(index=False)
    st.download_button("Download Clustered Data (CSV)", data=csv_data,
                       file_name="clustered_data.csv", mime="text/csv")

with tab3:
    st.header("🧩 Association Rules")
    # Bin numeric features into Low/Med/High categories
    df_assoc = df.copy()
    df_assoc['Age_Bin'] = pd.cut(df_assoc['Age'], bins=3, labels=['Age_Low','Age_Med','Age_High'])
    df_assoc['Income_Bin'] = pd.cut(df_assoc['Annual Income (k$)'], bins=3, labels=['Income_Low','Income_Med','Income_High'])
    df_assoc['Score_Bin'] = pd.cut(df_assoc['Spending Score (1-100)'], bins=3, labels=['Score_Low','Score_Med','Score_High'])
    # One-hot encode the binned categories
    ohe_df = pd.get_dummies(df_assoc[['Age_Bin','Income_Bin','Score_Bin']].astype(str))
    # Apply Apriori
    frequent_itemsets = apriori(ohe_df, min_support=0.1, use_colnames=True)
    if frequent_itemsets.empty:
        st.warning("No frequent itemsets found (support threshold may be too high).")
    else:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        if rules.empty:
            st.warning("No association rules found (confidence threshold may be too high).")
        else:
            rules_sorted = rules.sort_values(by='confidence', ascending=False).head(10)
            st.subheader("Top 10 Association Rules by Confidence")
            st.dataframe(rules_sorted[['antecedents','consequents','support','confidence','lift','leverage','conviction']])
            rules_csv = rules_sorted[['antecedents','consequents','support','confidence','lift','leverage','conviction']].to_csv(index=False)
            st.download_button("Download Rules (CSV)", data=rules_csv,file_name="association_rules.csv", mime="text/csv")
