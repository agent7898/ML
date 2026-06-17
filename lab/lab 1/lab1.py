import matplotlib
matplotlib.use('Qt5Agg')  
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
print(df)

columns = df.select_dtypes(include=['float64', 'int64']).columns

n_columns = 3
n_rows = 3

fig, axes = plt.subplots(n_rows , n_columns, figsize=(15,30)) 
axes = axes.flatten()
for i, column in enumerate(columns):
    ax = axes[i]
    df[column].hist(bins=30, edgecolor='black', grid=False, ax=ax)
    ax.set(title=f"Histogram of {column}", xlabel=column, ylabel="Frequency")
plt.tight_layout()
plt.savefig("Histogram_plots.png")
plt.show()


fig, axes = plt.subplots(n_rows , n_columns, figsize=(15,10))  
axes = axes.flatten()
for i, column in enumerate(columns):
    ax = axes[i]  
    df.boxplot(column=column, grid=False, ax=ax)
    ax.set(title=f"Box Plot of {column}")
plt.tight_layout()
plt.savefig("BOX_plots.png")
plt.show()

print("Outliers Detection:")
outliers_summary = {}
for feature in columns:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    outliers_summary[feature] = len(outliers)
    print(f"{feature}: {len(outliers)} outliers")