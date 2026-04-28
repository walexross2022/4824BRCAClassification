import pandas as pd

# COAD
print("=== COAD ===")
expr = pd.read_csv('H:/MLFinalProj/4824BRCAClassification/data/TCGA-COAD/coad_expression_matrix.csv', nrows=0)
sub = pd.read_csv('H:/MLFinalProj/4824BRCAClassification/data/TCGA-COAD/coad_subtypes.csv')
print("Features:", expr.shape[1] - 1)
print("Samples in expr:", expr.shape[1] - 1)  # placeholder
print("Subtypes:", sub['Subtype_Selected'].unique())
print("Subtype samples:", len(sub))

# PRAD
print("\n=== PRAD ===")
sub2 = pd.read_csv('H:/MLFinalProj/4824BRCAClassification/data/TCGA-PRAD/prad_subtypes.csv')
print("Subtypes:", sub2['Subtype_Selected'].unique())
print("Subtype samples:", len(sub2))

# BRCA
print("\n=== BRCA ===")
sub3 = pd.read_csv('H:/MLFinalProj/4824BRCAClassification/data/TCGA-BRCA/brca_subtypes.csv')
print("Subtypes:", sub3['Subtype_Selected'].unique())
print("Subtype samples:", len(sub3))
