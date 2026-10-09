# Experiments

Research notebooks for HCE-based cell-type classification. Every notebook starts with a
setup cell that `chdir`s to the repo root, so data paths (`lung.h5ad`, `census_data/`,
`lab-data/`, `All_cells.h5ad`) and output dirs (`*_results/`) are resolved relative to the
repo root — keep the datasets there, not next to the notebooks.

## `lung/` — single-tissue (lung) experiments

| Notebook | Purpose |
|---|---|
| `lung_dataset_explorer.ipynb` | Inspect `lung.h5ad` annotation columns |
| `investigate_hierarchy_structure.ipynb` | Diagnose and fix the multi-level hierarchy bug |
| `lung_hce_end_to_end_training_CORRECTED.ipynb` | **Canonical** hierarchy-fixed reference training |
| `lung_hce_final.ipynb`, `lung_hce_final_improved_v2.ipynb` | Option-A HCE training runs |
| `lung_hce_v3.ipynb` | Latest lung HCE iteration (writes `lung_hce_v3_results/`) |
| `lung_to_allcells_validation.ipynb` | Evaluate the v3 lung model on `All_cells.h5ad` |

## `multi_tissue/` — current multi-tissue work

| File | Purpose |
|---|---|
| `multi_tissue_hce_v11_comparison.ipynb` | **Latest** HCE vs CE comparison (uses `multi_tissue_v11_config.py`) |
| `multi_tissue_v11_config.py` | Shared config/helpers for v11 |
| `multi_tissue_hce_v10_comparison.ipynb` | Previous HCE vs CE comparison |
| `multi_tissue_lr_baseline.ipynb` | Logistic-regression baseline |
| `tier2_diagnostics.ipynb` | Dataset/label diagnostics |
| `census_data_explorer.ipynb` | Explore and export CELLxGENE census data into `census_data/` |

## `multi_tissue/archive/`

Superseded iterations `multi_tissue_hce.ipynb` (v1) through `v9`, kept for reference.
Higher version numbers are newer.
