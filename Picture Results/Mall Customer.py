import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Prepare output directory
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load and preprocess the dataset
def load_data():
    dataset_path = SCRIPT_DIR.parent / "DataSets" / "Mall_Customers.csv"
    df = pd.read_csv(dataset_path)
    df.columns = [col.strip() for col in df.columns]
    if 'Genre' in df.columns:
        df.rename(columns={'Genre': 'Gender'}, inplace=True)
    if 'CustomerID' in df.columns:
        df.drop('CustomerID', axis=1, inplace=True)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    return df

# Scale features to normalize the data
def scale_features(df):
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled

# Apply K-Means clustering and save plots
def apply_kmeans(df, X_scaled):
    inertias = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    # Elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "elbow_method_kmeans.png")
    plt.close()

    # KMeans with k=5
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    df['KMeans_Cluster'] = labels

    # Scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df,
                    x='Annual Income (k$)',
                    y='Spending Score (1-100)',
                    hue='KMeans_Cluster',
                    palette='tab10')
    plt.title('K-Means Clustering Result')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "kmeans_clusters.png")
    plt.close()

# Apply Single Linkage clustering and save dendrogram
def apply_single_linkage(df, X_scaled):
    linkage_matrix = linkage(X_scaled, method='single')
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Single Linkage Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "single_linkage_dendrogram.png")
    plt.close()

    df['SingleLink_Cluster'] = fcluster(linkage_matrix, t=5, criterion='distance')

# Apply Association Rules and print to console
def apply_association_rules(df):
    df['Income_Bin'] = pd.cut(df['Annual Income (k$)'],
                              bins=[0, 40, 70, 150],
                              labels=['Low', 'Medium', 'High'])
    df['Spending_Bin'] = pd.cut(df['Spending Score (1-100)'],
                                bins=[0, 40, 70, 100],
                                labels=['Low', 'Medium', 'High'])
    basket = pd.get_dummies(
        df[['KMeans_Cluster', 'SingleLink_Cluster', 'Income_Bin', 'Spending_Bin']].astype(str)
    ).astype(bool)

    freq_items = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(freq_items, metric='confidence', min_threshold=0.5)

    if not rules.empty:
        top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
        print("\nTop 10 Association Rules:")
        for _, row in top_rules.iterrows():
            antecedents = ', '.join(row['antecedents'])
            consequents = ', '.join(row['consequents'])
            print(f"Rule: IF {antecedents} THEN {consequents} "
                  f"| Support: {row['support']:.2f}, "
                  f"Confidence: {row['confidence']:.2f}, "
                  f"Lift: {row['lift']:.2f}")
    else:
        print("\nNo strong association rules found.")

# Main function
def main():
    print("Loading dataset...")
    df = load_data()

    print("Scaling features...")
    X_scaled = scale_features(df)

    print("Applying K-Means Clustering...")
    apply_kmeans(df, X_scaled)
    print(f"Saved K-Means plots in: {OUTPUT_DIR}")

    print("Applying Single Linkage Clustering...")
    apply_single_linkage(df, X_scaled)
    print(f"Saved hierarchical dendrogram in: {OUTPUT_DIR}")

    print("Applying Association Rules...")
    apply_association_rules(df)

    print("All processes completed successfully.")

if __name__ == "__main__":
    main()
