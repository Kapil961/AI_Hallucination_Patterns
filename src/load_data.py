import pandas as pd
import os

input_file = "data/raw/ai_hallucination_trials.jsonl"
output_file = "data/processed/ai_hallucination_trials.csv"

df = pd.read_json(input_file, lines=True)

os.makedirs("data/processed", exist_ok=True)

df.to_csv(output_file, index=False)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("CSV saved to:", output_file)

print("\nColumns:")
for column in df.columns:
    print("-", column)