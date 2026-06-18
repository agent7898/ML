import matplotlib 
matplotlib.use('Qt5Agg')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler

iris = load_iris() 
df = pd.DataFrame(iris.data, columns=iris.feature_names) 
print(iris.feature_names)

scaler = StandardScaler() 
scaled_data = scaler.fit_transform(df) 

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)


pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
print(pca_df)

plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=iris.target, cmap='viridis')
plt.title("PCA of Iris Dataset (2 components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Target')
plt.tight_layout()
plt.savefig("PCA_Iris.png")