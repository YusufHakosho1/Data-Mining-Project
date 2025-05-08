# Mall Customers Clustering Project (Console-Based)
# Dataset: Mall_Customers.csv (placed in working directory)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess the dataset
def load_data():
    # Load the dataset
    df = pd.read_csv('Mall_Customers.csv')
    
    # Clean the data: Strip column names and rename the 'Genre' column to 'Gender'
    df.columns = [col.strip() for col in df.columns]
    df.rename(columns={'Genre': 'Gender'}, inplace=True)
    
    # Drop the 'CustomerID' column as it's not needed for analysis
    df.drop('CustomerID', axis=1, inplace=True)
    
    # Convert 'Gender' to numerical values: Male = 0, Female = 1
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    return df

# Scale features to normalize the data
def scale_features(df):
    # Define the features to scale
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # Apply StandardScaler to normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    return X_scaled

# Apply K-Means clustering and visualize the elbow method to find the optimal number of clusters
def apply_kmeans(df, X_scaled):
    inertias = []
    
    # Try different numbers of clusters to find the optimal number using the Elbow method
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    # Plot the elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig("elbow_method_kmeans.png")  # Save elbow method plot
    
    # Perform K-Means clustering with 5 clusters based on the elbow method
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    df['KMeans_Cluster'] = labels

    # Plot the K-Means clustering results
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='KMeans_Cluster', palette='tab10')
    plt.title('K-Means Clustering Result')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.savefig("kmeans_clusters.png")  # Save K-Means clustering plot

# Apply Single Linkage clustering and visualize the dendrogram
def apply_single_linkage(df, X_scaled):
    # Perform single linkage hierarchical clustering
    linkage_matrix = linkage(X_scaled, method='single')
    
    # Plot the dendrogram for Single Linkage
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Single Linkage Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.savefig("single_linkage_dendrogram.png")  # Save dendrogram plot
    
    # Assign cluster labels to the dataset
    df['SingleLink_Cluster'] = fcluster(linkage_matrix, t=5, criterion='distance')

# Apply Association Rules using the Apriori algorithm
def apply_association_rules(df):
    # Bin the 'Annual Income' and 'Spending Score' into categories for association rules
    df['Income_Bin'] = pd.cut(df['Annual Income (k$)'], bins=[0, 40, 70, 150], labels=['Low', 'Medium', 'High'])
    df['Spending_Bin'] = pd.cut(df['Spending Score (1-100)'], bins=[0, 40, 70, 100], labels=['Low', 'Medium', 'High'])

    # Create a basket of binary values for association rules
    basket = pd.get_dummies(df[['KMeans_Cluster', 'SingleLink_Cluster', 'Income_Bin', 'Spending_Bin']].astype(str))
    basket = basket.astype(bool)
    
    # Apply the Apriori algorithm to find frequent itemsets with minimum support of 0.1
    freq_items = apriori(basket, min_support=0.1, use_colnames=True)
    
    # Generate association rules based on confidence of at least 0.5
    rules = association_rules(freq_items, metric='confidence', min_threshold=0.5)

    # If there are any strong association rules, display the top 10 based on confidence
    if not rules.empty:
        top_rules = rules.sort_values(by='confidence', ascending=False).head(10)
        print("\nTop 10 Association Rules:")
        for _, row in top_rules.iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            print(f"Rule: IF {antecedents} THEN {consequents} | Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")
    else:
        print("\nNo strong association rules found.")

# Main function to load data, apply algorithms, and visualize the results
def main():
    print("Loading dataset...")
    df = load_data()

    print("Scaling features...")
    X_scaled = scale_features(df)

    print("Applying K-Means Clustering...")
    apply_kmeans(df, X_scaled)

    print("Applying Single Linkage Clustering...")
    apply_single_linkage(df, X_scaled)

    print("Applying Association Rules...")
    apply_association_rules(df)

    print("All processes completed successfully.")

if __name__ == "__main__":
    main()
