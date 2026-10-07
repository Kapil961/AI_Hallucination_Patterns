import pandas as pd

file_path = "data/processed/clean_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset shape:")
print(df.shape)

print("\nModels:")
print(df["model"].value_counts())

print("\nHallucination by model:")
print(pd.crosstab(df["model"], df["hallucination"]))

print("\nHallucination rate by model:")
print(
    df.groupby("model")["hallucination"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nHallucination by condition:")
print(pd.crosstab(df["condition"], df["hallucination"]))

print("\nResponse length:")
print(df["response_length"].describe())

print("\nMissing values in important columns:")
important_columns = [
    "record_id",
    "condition",
    "model",
    "model_version",
    "prompt_string",
    "hallucination",
    "response_length"
]

print(df[important_columns].isnull().sum())