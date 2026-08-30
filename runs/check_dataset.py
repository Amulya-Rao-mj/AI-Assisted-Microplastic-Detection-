import pandas as pd

CSV_PATH = "Photos_data.xlsx - Sheet1.csv"

df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8-sig"
)

# Clean column names
df.columns = (
    df.columns
    .astype(str)
    .str.replace("\r", " ", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.strip()
)

print("\n========================================")
print("MORPHOLOGY VALUES IN YOUR DATASET")
print("========================================\n")

counts = df["Morphology of particle"].value_counts(
    dropna=False
)

print(counts.to_string())


print("\n========================================")
print("TOTAL ROWS")
print("========================================")

print(len(df))


print("\n========================================")
print("IMAGE FILE EXAMPLES")
print("========================================")

print(
    df[
        [
            "Image File",
            "Morphology of particle"
        ]
    ].head(30).to_string(index=False)
)