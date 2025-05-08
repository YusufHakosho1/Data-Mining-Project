import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

# Clustering and association imports
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import plotly.express as px

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Configure page for wide layout
st.set_page_config(page_title="Mall Customers Dashboard", layout="wide")
st.title("Mall Customers Segmentation & Analysis")

# --- Data Loading ---
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data("Mall_Customers.csv")
except FileNotFoundError:
    uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload 'Mall_Customers.csv'.")
        st.stop()

expected_cols = ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
if not all(col in df.columns for col in expected_cols):
    st.error(f"Data must include columns: {expected_cols}")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filter Customers")
genre_filter = st.sidebar.multiselect("Genre", df['Genre'].unique(), default=df['Genre'].unique())
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
inc_min, inc_max = float(df['Annual Income (k$)'].min()), float(df['Annual Income (k$)'].max())
income_range = st.sidebar.slider("Income Range (k$)", inc_min, inc_max, (inc_min, inc_max))
score_min, score_max = int(df['Spending Score (1-100)'].min()), int(df['Spending Score (1-100)'].max())
score_range = st.sidebar.slider("Score Range (1-100)", score_min, score_max, (score_min, score_max))

df_filtered = df[
    (df['Genre'].isin(genre_filter)) &
    (df['Age'].between(*age_range)) &
    (df['Annual Income (k$)'].between(*income_range)) &
    (df['Spending Score (1-100)'].between(*score_range))
].copy()
st.sidebar.write(f"Filtered: {len(df_filtered)} rows")

# Main tabs
tab_data, tab_clustering, tab_association, tab_prediction = st.tabs(
    ["Data Explorer", "Clustering", "Association Rules", "Predict Segment"]
)

# --- Data Explorer ---
with tab_data:
    st.header("Data Explorer")
    st.dataframe(df_filtered, height=300)
    st.download_button("Download Filtered CSV",
        df_filtered.to_csv(index=False).encode('utf-8'),
        file_name="filtered_customers.csv"
    )

# Prepare X for clustering
features = ['Age','Annual Income (k$)','Spending Score (1-100)']
X = df_filtered[features].values if len(df_filtered)>0 else np.empty((0,3))

# --- Clustering ---
with tab_clustering:
    st.header("Clustering Analysis")
    k_tab, h_tab = st.tabs(["K-Means", "Hierarchical"])

    # K-Means
    with k_tab:
        st.subheader("K-Means Clustering")
        if X.shape[0]<2:
            st.warning("Not enough data for K-Means.")
        else:
            k = st.slider("k (clusters)", 2, min(10,X.shape[0]), 3)
            km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
            df_km = df_filtered.copy()
            df_km['Cluster'] = (km.labels_+1).astype(str)

            # Multiselect clusters
            clusters = sorted(df_km['Cluster'].unique(), key=int)
            select_k = st.multiselect("Show Clusters", clusters, default=clusters)

            fig_k3d = px.scatter_3d(
                df_km[df_km['Cluster'].isin(select_k)],
                x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                color='Cluster', title=f"K-Means (k={k})",
                template='plotly_dark'
            )
            fig_k3d.update_traces(marker=dict(size=5,opacity=0.8))
            fig_k3d.update_layout(legend=dict(itemclick='toggle', itemdoubleclick='toggleothers'))
            st.plotly_chart(fig_k3d, use_container_width=True)

            # Export CSV
            st.download_button("Download K-Means Results",
                df_km.to_csv(index=False).encode('utf-8'),
                file_name="kmeans_clusters.csv"
            )

    # Hierarchical
    with h_tab:
        st.subheader("Hierarchical Clustering")
        if X.shape[0]<2:
            st.warning("Not enough data for Hierarchical.")
        else:
            method = st.selectbox("Linkage Method", ["ward","single","complete","average"])
            Z = linkage(X, method=method)
            thr = st.slider("Distance Threshold", 0.0, float(Z[:,2].max()), float(Z[:,2].max()/2))

            # Cluster labels
            labels = fcluster(Z, t=thr, criterion='distance')
            df_hc = df_filtered.copy()
            df_hc['Cluster'] = labels.astype(str)

            # Multiselect clusters
            clhs = sorted(df_hc['Cluster'].unique(), key=int)
            select_h = st.multiselect("Show Clusters", clhs, default=clhs)

            fig_h3d = px.scatter_3d(
                df_hc[df_hc['Cluster'].isin(select_h)],
                x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                color='Cluster', title=f"Hierarchical ({method})",
                template='plotly_dark'
            )
            fig_h3d.update_traces(marker=dict(size=5,opacity=0.8))
            fig_h3d.update_layout(legend=dict(itemclick='toggle', itemdoubleclick='toggleothers'))
            st.plotly_chart(fig_h3d, use_container_width=True)

            st.download_button("Download Hierarchical Results",
                df_hc.to_csv(index=False).encode('utf-8'),
                file_name="hierarchical_clusters.csv"
            )

# --- Association Rules ---
with tab_association:
    st.header("Association Rule Mining")
    c1,c2,c3 = st.columns(3)
    age_bins = c1.slider("Age bins",2,10,3)
    inc_bins = c2.slider("Income bins",2,10,4)
    score_bins = c3.slider("Score bins",2,10,3)
    min_s = st.slider("Min Support",0.01,1.0,0.1,0.01)
    min_c = st.slider("Min Confidence",0.01,1.0,0.5,0.01)

    if st.button("Generate Rules"):
        df_ar = df_filtered.copy()
        try:
            df_ar['Age_bin']=pd.qcut(df_ar['Age'],q=age_bins,duplicates='drop').astype(str)
            df_ar['Inc_bin']=pd.qcut(df_ar['Annual Income (k$)'],q=inc_bins,duplicates='drop').astype(str)
            df_ar['Score_bin']=pd.qcut(df_ar['Spending Score (1-100)'],q=score_bins,duplicates='drop').astype(str)
        except Exception as e:
            st.error(f"Binning error: {e}")
        else:
            txns = []
            for _,r in df_ar.iterrows():
                txns.append([f"Genre={r.Genre}",f"Age={r.Age_bin}",f"Inc={r.Inc_bin}",f"Score={r.Score_bin}"])
            te=TransactionEncoder()
            arr=te.fit(txns).transform(txns)
            df_te=pd.DataFrame(arr,columns=te.columns_)
            fi=apriori(df_te,min_support=min_s,use_colnames=True)
            if fi.empty:
                st.warning("No frequent itemsets.")
            else:
                rules=association_rules(fi,metric="confidence",min_threshold=min_c)
                if rules.empty:
                    st.warning("No rules found.")
                else:
                    # prettify rules
                    rules['Rule']=rules.apply(
                        lambda r: "If "+ " & ".join(sorted(r.antecedents))
                                  + " then " + " & ".join(sorted(r.consequents)),
                        axis=1
                    )
                    disp=rules[['Rule','support','confidence','lift']]
                    st.dataframe(disp)
                    st.download_button("Download Rules CSV",
                        disp.to_csv(index=False).encode('utf-8'),
                        file_name="association_rules.csv"
                    )

# --- Predict Segment ---
with tab_prediction:
    st.header("Predict New Customer Segment")
    k_pred = st.slider("k for prediction",2,min(10, X.shape[0] if X.size else 10),3)
    if X.shape[0]>=k_pred and X.shape[0]>0:
        km2=KMeans(n_clusters=k_pred,random_state=42,n_init='auto').fit(X)
        c1,c2,c3=st.columns(3)
        age_n=c1.number_input("Age",1,100,30)
        inc_n=c2.number_input("Income (k$)",1,500,50)
        score_n=c3.number_input("Score (1-100)",1,100,50)
        if st.button("Assign Cluster"):
            pt=np.array([[age_n,inc_n,score_n]])
            lab=int(np.argmin(np.linalg.norm(km2.cluster_centers_-pt,axis=1)))+1
            st.success(f"Assigned to cluster {lab}")
    else:
        st.warning("Adjust filters or k for prediction.")
