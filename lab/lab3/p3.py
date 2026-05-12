import matplotlib 
matplotlib.use('Agg')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris() 
df = pd.DataFrame(iris.data, columns=iris.feature_names) 
print(iris.feature_names)

# Standardize the features (important for PCA)
scaler = StandardScaler() 
scaled_data = scaler.fit_transform(df) 

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)


# Create a DataFrame for the 2 principal components
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
print(pca_df)
# Visualize the result
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=iris.target, cmap='viridis')
plt.title("PCA of Iris Dataset (2 components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Target')
plt.savefig("PCA_Iris.png")
if plt.get_backend().lower() != 'agg':
	plt.show()