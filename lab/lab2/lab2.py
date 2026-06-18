import matplotlib
matplotlib.use('Qt5Agg') 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing  

df = fetch_california_housing(as_frame=True).frame  

plt.subplots(figsize=(12, 12))  
plt.title("Correlation Matrix Heatmap") 
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=False, linewidths=.2)
plt.savefig("heatmap.png") 

sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal','Latitude']], diag_kind="kde")
plt.suptitle("Pairplots of relationship between variables")
plt.tight_layout()
plt.savefig("pairplots.png")