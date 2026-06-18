import pandas as pd

df = pd.read_csv('training_data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]   

def find_s_algorithm(X, y):
    hypothesis = None
    print("\n\nIntial Hypothesis is",hypothesis)
    for i in range(len(X)): 
        if y.iloc[i].strip().lower() == 'yes':  
            if hypothesis is None:
                hypothesis = X.iloc[i].tolist()  
            else:
                 for j in range(len(hypothesis)):
                    if hypothesis[j] != X.iloc[i, j]:
                        hypothesis[j] = '?' 
            print(f"h[{i}]=",hypothesis)
        else:
            print(f"h[{i}] Ignored as target concept is negative")
    return hypothesis


print("given dataset:\n",X)
hypothesis = find_s_algorithm(X, y)

print("\n\nHypothesis consistent with the positive examples:", hypothesis)
