import pandas as pd
import os

input_file = "data/processed/ai_hallucination_trials.csv"
output_file = "data/processed/clean_dataset.csv"

# Load dataset
df = pd.read_csv(input_file)

# Remove records that are not suitable for hallucination analysis
df = df[~df["hallucination_code"].isin(["TBD", "N/A"])].copy()

# Create binary hallucination column
df["hallucination"] = df["hallucination_code"].apply(
    lambda x: 1 if str(x).startswith("y") else 0
)

# Calculate response length in words
df["response_length"] = df["response"].fillna("").apply(
    lambda x: len(str(x).split())
)

# Select the important columns
clean_df = df[
    [
        "record_id",
        "condition",
        "model",
        "model_version",
        "date",
        "prompt_string",
        "hallucination_code",
        "hallucination_subcategory",
        "response",
        "hallucination",
        "response_length"
    ]
]

# Create output directory if needed
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
clean_df.to_csv(output_file, index=False)

print("Cleaning completed successfully!")
print("Rows:", len(clean_df))
print("Columns:", len(clean_df.columns))
print("Saved to:", output_file)

print("\nHallucination distribution:")
print(clean_df["hallucination"].value_counts())

print("\nHallucination percentage:")
print(clean_df["hallucination"].value_counts(normalize=True) * 100)