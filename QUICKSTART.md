# Cell2Sentence + HCE Integration - Quick Start Guide

## Summary of Work Completed

I've successfully integrated the Hierarchical Cross-Entropy (HCE) loss function from the HCE-HugoLab project into Cell2Sentence. Here's what was done:

### 1. Files Created

#### A. `src/cell2sentence/hce_trainer.py`
Custom HuggingFace Trainer that implements HCE loss:
- **HCETrainer class**: Extends Huggingface Trainer with hierarchical loss computation
- **build_reachability_matrix_from_hierarchy()**: Creates reachability matrix from nested hierarchy
- **build_reachability_matrix_from_ontology()**: Creates reachability matrix from parent-child mappings

#### B. `train_lung_hce.py`
Complete training script for lung.h5ad dataset:
- Loads and prepares the lung.h5ad dataset
- Builds cell type hierarchy (customizable)
- Trains Cell2Sentence model with HCE loss
- Ready to run once dependencies are installed

#### C. `hce_example_simple.py`
Minimal example showing how to use HCE loss:
- Simple demonstration of the key code changes needed
- Good starting point for understanding the integration

#### D. `HCE_INTEGRATION_README.md`
Comprehensive documentation covering:
- What HCE loss is and why it's useful
- How to use it with Cell2Sentence
- Reachability matrix explanation
- Configuration options
- Examples and best practices

### 2. Files Modified

#### `src/cell2sentence/csmodel.py`
Extended the `fine_tune()` method to support custom trainers:
- Added `trainer_class` parameter (default: standard Trainer)
- Added `trainer_kwargs` parameter for custom trainer arguments
- **Fully backward compatible** - existing code will work unchanged

## Environment Setup

**Virtual Environment**: `c2s-justin` (Python 3.13)
- Located at: `/home/hugolab/Documents/Cell2Sentence-HugoLab/Cell2Sentence-HugoLab/c2s-justin`
- Installation in progress (pip install -e .)

## Quick Start

### 1. Activate Environment

```bash
cd /home/hugolab/Documents/Cell2Sentence-HugoLab/Cell2Sentence-HugoLab
source c2s-justin/bin/activate
```

### 2. Run Simple Example

```bash
python hce_example_simple.py
```

This will demonstrate the reachability matrix construction.

### 3. Train on Lung Dataset

```bash
python train_lung_hce.py
```

This will:
- Load lung.h5ad
- Create a simple cell type hierarchy
- Train a Cell2Sentence model with HCE loss
- Save the trained model to `./outputs/c2s_lung_hce_trained/`

## Key Code Pattern

Here's the minimal code change to use HCE loss:

```python
from cell2sentence.hce_trainer import HCETrainer, build_reachability_matrix_from_ontology
import torch

# Define hierarchy
ontology_dict = {"child_type": "parent_type", ...}
all_types = ["parent_type", "child_type", ...]

# Build reachability matrix
reachability_matrix = build_reachability_matrix_from_ontology(ontology_dict, all_types)

# Configure HCE trainer
trainer_kwargs = {
    'reachability_matrix': torch.tensor(reachability_matrix, dtype=torch.float32),
    'use_hce_loss': True
}

# Train with HCE loss
csmodel.fine_tune(
    csdata=csdata_obj,
    task='cell_type_prediction',
    train_args=training_args,
    trainer_class=HCETrainer,      # Add this
    trainer_kwargs=trainer_kwargs   # Add this
)
```

## What is HCE Loss?

HCE (Hierarchical Cross-Entropy) loss leverages the hierarchical structure of cell types:

**Traditional Cross-Entropy**:
- Treats all classes as independent
- Misclassifying "CD4 T cell" as "B cell" is equally bad as misclassifying it as "CD8 T cell"

**HCE Loss**:
- Accounts for biological relationships
- Misclassifying "CD4 T cell" as "CD8 T cell" (both T cells) is less penalized than misclassifying it as "B cell"
- Propagates probabilities through the hierarchy using a reachability matrix

**Result**: More biologically informed model that respects cell type ontology.

## Customization

### Using a Proper Cell Ontology

For production use, replace the simple hierarchy in `train_lung_hce.py` with:

```python
# Load from Cell Ontology
from cellnet.utils.cell_ontology import CellOntology

ontology = CellOntology()
# Use ontology to build proper hierarchy
```

### Adjusting Training

Modify `train_lung_hce.py` to adjust:
- Model size: Change `MODEL_NAME` (e.g., 'EleutherAI/pythia-410m' for larger model)
- Training epochs: Change `num_train_epochs`
- Batch size: Adjust `per_device_train_batch_size`
- Top K genes: Change `TOP_K_GENES`

## File Locations

All files are in: `/home/hugolab/Documents/Cell2Sentence-HugoLab/Cell2Sentence-HugoLab/`

- **Source code**: `src/cell2sentence/hce_trainer.py`
- **Training script**: `train_lung_hce.py`
- **Simple example**: `hce_example_simple.py`
- **Full documentation**: `HCE_INTEGRATION_README.md`
- **Dataset**: `lung.h5ad`
- **Virtual env**: `c2s-justin/`

## Next Steps

1. **Wait for installation to complete** - Check with:
   ```bash
   source c2s-justin/bin/activate
   pip list | grep cell2sentence
   ```

2. **Test the simple example**:
   ```bash
   python hce_example_simple.py
   ```

3. **Run training on lung.h5ad**:
   ```bash
   python train_lung_hce.py
   ```

4. **Customize hierarchy** - Edit `train_lung_hce.py` to use your specific cell type ontology

5. **Monitor training** - Use TensorBoard:
   ```bash
   tensorboard --logdir outputs/logs
   ```

## Troubleshooting

### If dataset format is different:
Check what columns are available in lung.h5ad:
```python
import scanpy as sc
adata = sc.read_h5ad('lung.h5ad')
print(adata.obs.columns)
```
Then modify `train_lung_hce.py` to use the correct column name.

### If you need a different model:
Change `MODEL_NAME` in `train_lung_hce.py` to any HuggingFace model or existing C2S checkpoint.

### If GPU memory is limited:
- Reduce batch size
- Use gradient checkpointing
- Use smaller model (e.g., pythia-160m instead of pythia-410m)

## References

- **HCE Repository**: `/home/hugolab/Documents/HCE/HCE-HugoLab/hce-classification`
- **Cell2Sentence Repository**: `/home/hugolab/Documents/Cell2Sentence-HugoLab/Cell2Sentence-HugoLab`
- **HCE Paper**: https://www.biorxiv.org/content/10.1101/2025.04.23.650210
- **Cell2Sentence Paper**: https://www.biorxiv.org/content/10.1101/2025.04.14.648850

---

**Created**: February 10, 2026
**Environment**: c2s-justin (Python 3.13)
**Status**: Ready to use once dependencies installation completes
