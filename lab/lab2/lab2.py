import matplotlib
matplotlib.use('Agg')  # Use Agg backend for saving files without display
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing  

# Load the dataset 
df = fetch_california_housing(as_frame=True).frame  

# Set up the grid layout for plots
plt.subplots(figsize=(12, 12))  
plt.title("Correlation Matrix Heatmap") 
# Heatmap of correlation matrix (Top plot) 
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=False, linewidths=.2)
plt.savefig("heatmap.png") 


# Pair plot for selected features (Bottom plot) 
sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal','Latitude']], diag_kind="kde")
plt.suptitle("Pairplots of relationship between variables")
plt.tight_layout()
plt.savefig("pairplots.png")
#plt.subplots_adjust(hspace=0.4)  # Adjust space between subplots  

# Show the combined figure 
