import pandas as pd

df = pd.read_csv("TCGA-BRCA.star_counts.tsv.gz", sep="\t", index_col=0)

print("Original shape:", df.shape)

df = df.T

print("After transpose:", df.shape)

print(df.head())

labels = pd.read_csv("brca_tcga_pan_can_atlas_2018_clinical_data.tsv", sep="\t")
print(labels.head())

labels = labels[["Sample ID", "Subtype"]]
labels = labels.dropna()

labels.columns = ["sample_id", "subtype"]

print(labels.head())

# Add sample_id column to expression data
df["sample_id"] = df.index

# Standardize IDs to patient-level (first 12 characters)
df["sample_id"] = df.index.str[:12]
labels["sample_id"] = labels["sample_id"].str[:12]

# Merge
df = df.merge(labels, on="sample_id")

print("After merge:", df.shape)
print("Unique subtypes:", df["subtype"].unique())
print("Counts:\n", df["subtype"].value_counts())

df = df[df["subtype"] != "BRCA_Normal"]

X = df.drop(columns=["sample_id", "subtype"])
y = df["subtype"]

print("Final shape:", X.shape)
print(y.value_counts())