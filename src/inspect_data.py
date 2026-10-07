import pandas as pd

file_path = "data/processed/ai_hallucination_trials.csv"

df = pd.read_csv(file_path)

print("Dataset shape:")
print(df.shape)

print("\nUnique models:")
print(df["model"].value_counts())

print("\nUnique hallucination codes:")
print(df["hallucination_code"].value_counts(dropna=False))

print("\nUnique hallucination subcategories:")
print(df["hallucination_subcategory"].value_counts(dropna=False))

print("\nUnique conditions:")
print(df["condition"].value_counts(dropna=False))

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 3 records:")
print(df.head(3).to_string())