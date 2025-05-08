import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit App Title
st.title("Mall Customers Data Analysis App")

# Sidebar - File Upload
st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display raw data
    st.subheader("Raw Dataset Preview")
    st.write(df.head())
    
    # Rename 'Genre' to 'Gender' if exists
    if 'Genre' in df.columns:
        df = df.rename(columns={'Genre': 'Gender'})
    # Map Gender values
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    # Drop 'CustomerID' if present
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
        
    # Feature Normalization
    st.subheader("Feature Normalization")
    scaler = MinMaxScaler()
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    # Ensure the numeric columns exist in the data
    for col in numeric_cols:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in the dataset.")
    df_norm = df.copy()
    df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])
    st.write("Sample of normalized features:")
    st.write(df_norm[numeric_cols].head())
    
    # K-Means Clustering
    st.subheader("K-Means Clustering")
    # Elbow plot for k = 1 to 10
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_norm[numeric_cols])
        inertia.append(kmeans.inertia_)
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(K, inertia, marker='o')
    ax_elbow.set_xlabel('Number of clusters k')
    ax_elbow.set_ylabel('Inertia')
    ax_elbow.set_title('Elbow Method For Optimal k')
    st.pyplot(fig_elbow)
    
    # Sidebar - select k for K-Means
    st.sidebar.header("K-Means Options")
    k = st.sidebar.slider("Select number of clusters (k) for K-Means", min_value=1, max_value=10, value=3, step=1)
    # Apply KMeans with selected k
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_norm[numeric_cols])
    df['KMeans_Cluster'] = kmeans.labels_
    # Scatter plot Annual Income vs Spending Score
    fig_cluster = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                             color=df['KMeans_Cluster'].astype(str), 
                             title=f'K-Means Clusters (k={k})')
    st.plotly_chart(fig_cluster)
    
    # Single Linkage Hierarchical Clustering
    st.subheader("Hierarchical Clustering (Single Linkage)")
    # Compute hierarchical linkage
    Z = linkage(df_norm[numeric_cols], method='single')
    # Display dendrogram
    fig_dendro, ax_dendro = plt.subplots(figsize=(8, 4))
    dendrogram(Z, ax=ax_dendro, orientation='top', distance_sort='descending', show_leaf_counts=False)
    ax_dendro.set_xlabel("Data Points")
    ax_dendro.set_ylabel("Distance")
    ax_dendro.set_title("Dendrogram (Single Linkage)")
    st.pyplot(fig_dendro)
    
    # Sidebar - select number of clusters for hierarchical
    st.sidebar.header("Hierarchical Options")
    num_clusters = st.sidebar.slider("Number of clusters for Hierarchical", min_value=2, max_value=10, value=3, step=1)
    # Assign cluster labels
    df['Hierarchical_Cluster'] = fcluster(Z, t=num_clusters, criterion='maxclust')
    # Display assigned cluster labels (first few entries)
    st.write("Cluster assignments (Hierarchical):")
    st.write(df[['Hierarchical_Cluster']].head(10))
    
    # Association Rules Mining
    st.subheader("Association Rules")
    # Binning Income and Spending Score into Low/Medium/High
    df_binned = df.copy()
    df_binned['IncomeCat'] = pd.cut(df_binned['Annual Income (k$)'], 3, labels=['Low Income', 'Medium Income', 'High Income'])
    df_binned['ScoreCat'] = pd.cut(df_binned['Spending Score (1-100)'], 3, labels=['Low Score', 'Medium Score', 'High Score'])
    # Prepare transactions list
    transactions = []
    for idx, row in df_binned.iterrows():
        items = []
        if pd.notna(row['IncomeCat']):
            items.append(str(row['IncomeCat']))
        if pd.notna(row['ScoreCat']):
            items.append(str(row['ScoreCat']))
        transactions.append(items)
    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_te = pd.DataFrame(te_ary, columns=te.columns_)
    # Apply Apriori
    frequent_itemsets = apriori(df_te, min_support=0.1, use_colnames=True)
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    # Sort by confidence and take top 10
    rules = rules.sort_values(by='confidence', ascending=False).head(10).copy()
    # Format antecedents and consequents for readability
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    # Display rules
    if not rules.empty:
        st.write("Top 10 association rules (by confidence):")
        st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        st.write("No association rules found with the given thresholds.")
else:
    st.write("Awaiting CSV file to be uploaded.")