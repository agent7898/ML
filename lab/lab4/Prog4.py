import pandas as pd
# Load the dataset
df = pd.read_csv('training_data.csv')
# Assume the last column is the class (target variable)
X = df.iloc[:, :-1]  # :-1 ->Features (all columns except the last), : -> take all rows leaving header row
y = df.iloc[:, -1]   # Class (the last column) -1-> selects only last column
# Find-S algorithm
def find_s_algorithm(X, y):
    # Initialize the hypothesis to None
    hypothesis = None
    print("Intial Hypothesis is",hypothesis)
    for i in range(len(X)): 
        if y.iloc[i].strip().lower() == 'yes':  # Only consider positive examples
            if hypothesis is None:
                hypothesis = X.iloc[i].tolist()  # Initialize with first positive example
            else:
                 for j in range(len(hypothesis)):
                    if hypothesis[j] != X.iloc[i, j]:
                        hypothesis[j] = '?'  # Generalize if values differ
            print(f"h[{i}]=",hypothesis)
        else:
            print(f"h[{i}] Ignored as target concept is negative")
    return hypothesis
# Get the most specific hypothesis

print("given dataset:\n",X)
hypothesis = find_s_algorithm(X, y)
# Output the hypothesis
print("Hypothesis consistent with the positive examples:", hypothesis)
