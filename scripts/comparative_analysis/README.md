# Comparative Analysis
`disease_code_check.ipynb` is used to check the ICD-PHECODE mapping

- **Input**: 
  - target disease name and ICD10 code(regular expression)
  - trained all-disease network
  - icd10-phecode mapping file and input-phecode mapping file
- **Output**:
  - disease-specific network
  - disease-specific auc/cindex