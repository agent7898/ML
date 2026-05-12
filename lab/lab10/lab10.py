import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data1 = load_breast_cancer(as_frame=True)
X = data1.data          
y = data1.target        

df = X.copy()
df['Diagnosis'] = y.map({0: 'malignant', 1: 'benign'})

print("Dataset shape:", df.shape)
print(df[['Diagnosis']].value_counts())

features = data1.feature_names
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

k = 2  
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]


plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2',hue='Cluster',style='Diagnosis',data=df, palette='viridis', s=70)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1],c='red', s=200, marker='X', label='Centroids')
plt.title('K-Means Clustering vs. True Labels (PCA-reduced)')
plt.legend()
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig("lab10.png")
plt.show()


print("Scaled-data Cluster Centers (2D PCA space):\n", centers)
print(f"\nExplained variance by 2 PCA components: {pca.explained_variance_ratio_.sum():.2%}")


contingency = pd.crosstab(df['Cluster'], df['Diagnosis'],rownames=['Cluster'], colnames=['Diagnosis'])
print("\nCluster vs Diagnosis:\n", contingency)