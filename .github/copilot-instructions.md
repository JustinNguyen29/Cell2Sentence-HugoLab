# Copilot Instructions for Cell2Sentence + HCE Workspace

## Workspace Structure

Two related research repos live side-by-side:
- **`Cell2Sentence-HugoLab/`** — LLM framework for single-cell RNA-seq (main codebase)
- **`HCE-HugoLab/hce-classification/`** — Hierarchical Cross-Entropy loss paper code (scTab-derived)

The Cell2Sentence repo has integrated HCE loss from HCE-HugoLab into `src/cell2sentence/hce_trainer.py`.

## Core Abstraction: Cell Sentences

Cells are transformed into **cell sentences**: space-separated gene names ordered by *descending expression*. This is the key representation that allows LLMs to process scRNA-seq data as text. Gene names are stored **UPPERCASE** throughout the codebase (`utils.py:generate_vocabulary`).

## Data Flow

```
AnnData (.h5ad)
  → CSData.adata_to_arrow()        # ranks genes, builds HF arrow dataset
  → C2SPromptFormatter.format_hf_ds()  # wraps sentences in task prompts
  → tokenize_loss_on_response()    # tokenizes; -100 masks prompt tokens from loss
  → HCETrainer / Trainer           # HuggingFace training loop
```

Key files:
- [`src/cell2sentence/csdata.py`](src/cell2sentence/csdata.py) — `CSData`: AnnData → on-disk arrow dataset
- [`src/cell2sentence/csmodel.py`](src/cell2sentence/csmodel.py) — `CSModel`: model loading, `fine_tune()`, embeddings/generation
- [`src/cell2sentence/prompt_formatter.py`](src/cell2sentence/prompt_formatter.py) — `C2SPromptFormatter`: JSON prompt templates
- [`src/cell2sentence/hce_trainer.py`](src/cell2sentence/hce_trainer.py) — `HCETrainer`: hierarchical loss via reachability matrix
- [`src/cell2sentence/tasks.py`](src/cell2sentence/tasks.py) — high-level functions (`generate_cells_conditioned_on_cell_type`, `predict_cell_types`)

## HCE Loss Integration Pattern

To use hierarchical loss, pass `trainer_class` and `trainer_kwargs` to `fine_tune()`:
```python
from cell2sentence.hce_trainer import HCETrainer, build_reachability_matrix_from_ontology
reachability_matrix = build_reachability_matrix_from_ontology(ontology_dict, all_cell_types)
csmodel.fine_tune(
    csdata=csdata_obj, task='cell_type_prediction', train_args=training_args,
    trainer_class=HCETrainer,
    trainer_kwargs={'reachability_matrix': torch.tensor(reachability_matrix, dtype=torch.float32), 'use_hce_loss': True}
)
```
The `fine_tune()` method is fully backward-compatible (`trainer_class=None` uses standard `Trainer`).

## Reachability Matrix Rules

- Shape: `(num_classes, num_classes)`; element `(i,j)=1` if class `j` is reachable from class `i` (i.e., `j` is `i` or a descendant)
- **Use vectorized matmul** for the hierarchical probability propagation — never loop over batch:
  ```python
  label_reachability = self.reachability_matrix[labels]  # (batch, num_classes)
  reachable_probs = (probs * label_reachability).sum(dim=1)
  hce_loss = -torch.log(torch.clamp(reachable_probs, min=1e-9)).mean()
  ```

## Hierarchy Building: Critical Bug Fix

When traversing multi-level cell annotation columns (e.g. `level_1_cell_type`, `level_2_cell_type`…), **stop at the first invalid value** — do not continue traversal. "Unknown", "None", "NA", NaN are not cell types. Failure to stop caused 93% of parent-child relationships to be broken in early training runs. The canonical fixed notebook is [`lung_hce_end_to_end_training_CORRECTED.ipynb`](lung_hce_end_to_end_training_CORRECTED.ipynb).

```python
def is_valid_annotation(value):
    if pd.isna(value): return False
    return str(value).strip().lower() not in ['unknown', 'none', 'na', 'n/a', 'nan', '']

for level_col in level_columns:
    value = row[level_col]
    if is_valid_annotation(value):
        valid_path.append((level_col, value))
    else:
        break  # stop — don't continue deeper
```

## Prompt Templates

Prompts live in [`src/cell2sentence/prompts/`](src/cell2sentence/prompts/) as JSON files. Each task has a list of `model_input` templates and `response` templates; one is **randomly selected per sample** during formatting. Templates use `{cell_sentence}`, `{num_genes}`, `{organism}`, `{cell_type}` placeholders.

Supported tasks in `C2SPromptFormatter`: `cell_type_prediction`, `cell_type_generation`. Multicell tasks (`tissue_prediction`, `natural_language_interpretation`) have their own formatters.

## Environment & Developer Workflow

```bash
# Activate local venv (Python 3.13)
source c2s-justin/bin/activate

# Install package in editable mode
make install          # runs: pip install -e .

# Run tests
make test             # runs: pytest src/cell2sentence/tests

# Run lint
make lint             # runs: pylint cell2sentence

# Build docs
make html             # sphinx-build into docs/build/
```

The venv is at `c2s-justin/` (not a conda env). The package is in `src/` layout; `setup.cfg` declares `package_dir = src`.

## Lung Dataset Experiments

Primary dataset: `lung.h5ad`. Experiment outputs land in:
- `lung_hce_balanced_results/` — balanced multi-split classification
- `lung_hce_end_to_end_results_corrected/` — end-to-end with hierarchy fix applied

Use `MIN_CELLS_PER_TYPE` / `MAX_CELLS_PER_TYPE` constants in notebooks to control class balancing. Filter unknown cells before balancing:
```python
mask = ~adata.obs[ANN_COL].astype(str).isin(['Unknown', 'unknown', 'UNKNOWN', 'nan', 'NA'])
adata = adata[mask].copy()
```
