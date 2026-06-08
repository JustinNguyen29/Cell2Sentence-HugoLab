# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is the **HugoLab fork of Cell2Sentence (C2S)** — an LLM framework for single-cell RNA-seq. The upstream package (`src/cell2sentence/`) is augmented here with **Hierarchical Cross-Entropy (HCE) loss** integration and a large body of research notebooks for lung and multi-tissue cell-type classification experiments.

## Core abstraction: cell sentences

Cells are transformed into **cell sentences** — space-separated gene names ordered by *descending expression*. This text representation is what lets LLMs natively process scRNA-seq data. Gene names are stored **UPPERCASE** throughout (`utils.py:generate_vocabulary`).

## Data flow

```
AnnData (.h5ad)
  → CSData.adata_to_arrow()            # ranks genes, builds on-disk HF arrow dataset
  → C2SPromptFormatter.format_hf_ds()  # wraps sentences in task prompts (random template per sample)
  → tokenize_loss_on_response()        # tokenizes; -100 masks prompt tokens from loss
  → HCETrainer / Trainer               # HuggingFace training loop
```

Key package files (`src/cell2sentence/`):
- `csdata.py` — `CSData`: AnnData → on-disk arrow dataset (`adata_to_arrow`, `csdata_from_arrow`, `csdata_from_multiple_arrow_datasets`)
- `csmodel.py` — `CSModel`: model loading, `fine_tune()`, `embed_cells_batched()`, `generate_from_prompt_batched()`
- `prompt_formatter.py` — `C2SPromptFormatter`: JSON prompt templates
- `hce_trainer.py` — `HCETrainer` plus reachability-matrix builders
- `tasks.py` — high-level entry points (`generate_cells_conditioned_on_cell_type`, `predict_cell_types_of_data`, `embed_cells`)
- `prompts/*.json` — prompt templates; each task has `model_input` and `response` lists, one randomly selected per sample. Placeholders: `{cell_sentence}`, `{num_genes}`, `{organism}`, `{cell_type}`. Supported single-cell tasks: `cell_type_prediction`, `cell_type_generation`; multicell tasks (`tissue_prediction`, `natural_language_interpretation`) have separate formatters.

## HCE loss integration

To use hierarchical loss, pass `trainer_class` and `trainer_kwargs` to `CSModel.fine_tune()` — it is fully backward-compatible (`trainer_class=None` uses the standard HF `Trainer`):

```python
from cell2sentence.hce_trainer import HCETrainer, build_reachability_matrix_from_ontology
reachability_matrix = build_reachability_matrix_from_ontology(ontology_dict, all_cell_types)
csmodel.fine_tune(
    csdata=csdata_obj, task='cell_type_prediction', train_args=training_args,
    trainer_class=HCETrainer,
    trainer_kwargs={'reachability_matrix': torch.tensor(reachability_matrix, dtype=torch.float32),
                    'use_hce_loss': True},
)
```

**Reachability matrix:** shape `(num_classes, num_classes)`; element `(i,j)=1` if class `j` is reachable from class `i` (i.e. `j` is `i` or a descendant). Always use **vectorized matmul** for hierarchical probability propagation — never loop over the batch:

```python
label_reachability = self.reachability_matrix[labels]   # (batch, num_classes)
reachable_probs = (probs * label_reachability).sum(dim=1)
hce_loss = -torch.log(torch.clamp(reachable_probs, min=1e-9)).mean()
```

HCE gives partial credit when the model predicts an ancestor of the true class, so biologically meaningful confusion is not penalized.

## Hierarchy building: critical bug fix

When traversing multi-level annotation columns (`level_1_cell_type`, `level_2_cell_type`, …), **stop at the first invalid value** — do not continue. "Unknown"/"None"/"NA"/NaN are not cell types. Not stopping previously broke 93% of parent-child relationships.

```python
def is_valid_annotation(value):
    if pd.isna(value): return False
    return str(value).strip().lower() not in ['unknown', 'none', 'na', 'n/a', 'nan', '']

for level_col in level_columns:
    value = row[level_col]
    if is_valid_annotation(value):
        valid_path.append((level_col, value))
    else:
        break  # stop — don't go deeper
```

Also filter unknown cells before class balancing, and use `MIN_CELLS_PER_TYPE` / `MAX_CELLS_PER_TYPE` constants in notebooks to control balancing:

```python
mask = ~adata.obs[ANN_COL].astype(str).isin(['Unknown', 'unknown', 'UNKNOWN', 'nan', 'NA'])
adata = adata[mask].copy()
```

## Developer workflow

The environment is a **local venv at `c2s-justin/` (Python 3.13), not conda**. The package uses a `src/` layout (`setup.cfg` declares `package_dir = src`).

```bash
source c2s-justin/bin/activate    # activate venv
make install                      # pip install -e .
make test                         # pytest src/cell2sentence/tests
pytest src/cell2sentence/tests/test_tasks.py::TestName    # single test
make lint                         # pylint cell2sentence
make html                         # sphinx docs → docs/build/
```

## Notebooks & experiments

Most active work happens in **Jupyter notebooks at the repo root**, not in the package. They are versioned by filename suffix (`multi_tissue_hce_v2.ipynb` … `v10`, `lung_hce_*`); higher version numbers are newer iterations. The canonical hierarchy-fixed reference notebook is `lung_hce_end_to_end_training_CORRECTED.ipynb`.

Experiment outputs land in `*_results/` directories (e.g. `multi_tissue_v9_results/`, `lung_hce_end_to_end_results_corrected/`). These directories, large `.h5ad` datasets, and `*.pt`/`*.pth` model files are git-ignored — do not commit them. The primary datasets (`lung.h5ad`, `brain*.h5ad`, `All_cells.h5ad`, `census_data/`, `lab-data/`) are large local files excluded from git.

## Standalone docs in this repo

Several Markdown files document the HCE work and notebook reviews; consult them before changing related code: `HCE_INTEGRATION_README.md`, `HIERARCHY_FIX_SUMMARY.md`, `HIERARCHY_FIX_QUICK_REF.md`, `BIOLOGICAL_SIGNIFICANCE.md`, `NOTEBOOK_ANALYSIS.md`, `QUICKSTART.md`.
