import pandas as pd

# STEP 1: Load Expression Data
df = pd.read_csv("TCGA-BRCA.star_counts.tsv.gz", sep="\t", index_col=0)
print("Original shape (genes x samples):", df.shape)

# Transpose to (samples x genes)
df = df.T
print("After transpose (samples x genes):", df.shape)

# STEP 2: Load Clinical Labels
labels = pd.read_csv("brca_tcga_pan_can_atlas_2018_clinical_data.tsv", sep="\t")

# Keep only relevant columns
labels = labels[["Sample ID", "Subtype"]].dropna()
labels.columns = ["sample_id", "subtype"]

print("\nLabel preview:")
print(labels.head())

# STEP 3: Align Sample IDs
# Convert both to patient-level IDs (first 12 characters)
df["sample_id"] = df.index.str[:12]
labels["sample_id"] = labels["sample_id"].str[:12]


# STEP 4: Merge Data
df = df.merge(labels, on="sample_id")

print("\nAfter merge:", df.shape)
print("Class distribution:\n", df["subtype"].value_counts())

# STEP 5: Clean Data
# Remove small/rare class
df = df[df["subtype"] != "BRCA_Normal"]

print("\nAfter removing BRCA_Normal:")
print(df["subtype"].value_counts())

# STEP 6: Final Dataset
X = df.drop(columns=["sample_id", "subtype"])
y = df["subtype"]

print("\nFinal feature matrix:", X.shape)
print("Final labels:\n", y.value_counts())
