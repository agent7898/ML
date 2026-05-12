import matplotlib
matplotlib.use('Qt5Agg')  # Use TkAgg backend
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the California Housing Dataset
data = fetch_california_housing(as_frame=True)

df = data.frame
print(df)
# Select numeric columns
columns = df.select_dtypes(include=['float64', 'int64']).columns
len(columns)
print (len(columns))
# Set up grid layout (two rows: one for histograms, one for box plots)
n_columns = 3
n_rows = (len(columns) + n_columns - 1) // n_columns  # Ensure enough rows

fig, axes = plt.subplots(n_rows , n_columns, figsize=(15,30))  # Double rows
axes = axes.flatten()
# Plot histograms
for i, column in enumerate(columns):
    ax = axes[i]
    df[column].hist(bins=30, edgecolor='black', grid=False, ax=ax)
    ax.set(title=f"Histogram of {column}", xlabel=column, ylabel="Frequency")
plt.tight_layout()
plt.savefig("Histogram_plots.png")
plt.show()

# Plot Boxplot
fig, axes = plt.subplots(n_rows , n_columns, figsize=(15,10))  # Double rows
axes = axes.flatten()
for i, column in enumerate(columns):
    ax = axes[i]  # Now, correctly indexing box plots
    df.boxplot(column=column, grid=False, ax=ax)
    ax.set(title=f"Box Plot of {column}")
# Adjust layout and display
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