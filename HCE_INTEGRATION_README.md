# HCE Loss Integration for Cell2Sentence

## Overview

This document describes the integration of Hierarchical Cross-Entropy (HCE) loss from the HCE-HugoLab project into the Cell2Sentence framework for improved cell type classification.

## What is HCE Loss?

Hierarchical Cross-Entropy (HCE) loss is a specialized loss function that leverages the inherent hierarchical structure within classification problems. Unlike standard cross-entropy which treats each class independently, HCE accounts for parent-child relationships between classes using a **reachability matrix**.

### Key Benefits:
- Improves model performance on hierarchical classification tasks
- Accounts for biological relationships between cell types
- Reduces misclassification errors by considering the cell type ontology

## Implementation

### New Files Created:

1. **`src/cell2sentence/hce_trainer.py`**: Custom HuggingFace Trainer implementation
   - `HCETrainer`: Custom trainer class that computes HCE loss instead of standard cross-entropy
   - `build_reachability_matrix_from_hierarchy()`: Builds reachability matrix from nested hierarchy
   - `build_reachability_matrix_from_ontology()`: Builds reachability matrix from parent-child dictionary

2. **`train_lung_hce.py`**: Training script for lung.h5ad dataset with HCE loss
   - Loads lung.h5ad dataset
   - Creates simple cell type hierarchy (can be customized with proper ontology)
   - Trains Cell2Sentence model with HCE loss

### Modified Files:

1. **`src/cell2sentence/csmodel.py`**: Extended `fine_tune()` method
   - Added `trainer_class` parameter to accept custom trainer classes
   - Added `trainer_kwargs` parameter to pass additional arguments to custom trainers
   - Maintains backward compatibility with existing code

## How to Use

### Basic Usage:

```python
import torch
from cell2sentence.csmodel import CSModel
from cell2sentence.csdata import CSData
from cell2sentence.hce_trainer import HCETrainer, build_reachability_matrix_from_ontology
from transformers import TrainingArguments

# 1. Define cell type hierarchy
# Simple parent-child mapping
ontology_dict = {
    "T cell": "Immune Cell",
    "B cell": "Immune Cell",
    "NK cell": "Immune Cell",
    # ... more mappings
}

# All cell types (including parent types)
all_cell_types = ["Immune Cell", "T cell", "B cell", "NK cell", ...]

# 2. Build reachability matrix
reachability_matrix = build_reachability_matrix_from_ontology(
    ontology_dict, 
    all_cell_types
)

# 3. Prepare your data (CSData object)
csdata_obj = CSData.adata_to_arrow(
    adata,
    label_col_names=['cell_type']
)

# 4. Initialize model
csmodel = CSModel(
    model_name_or_path='EleutherAI/pythia-160m',
    save_dir='./outputs',
    save_name='my_model'
)

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir='./outputs/trained_model',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    # ... other arguments
)

# 6. Train with HCE loss
trainer_kwargs = {
    'reachability_matrix': torch.tensor(reachability_matrix, dtype=torch.float32),
    'use_hce_loss': True
}

csmodel.fine_tune(
    csdata=csdata_obj,
    task='cell_type_prediction',
    train_args=training_args,
    trainer_class=HCETrainer,  # Use HCE trainer instead of default
    trainer_kwargs=trainer_kwargs
)
```

### Running the Lung Dataset Example:

```bash
# Activate the virtual environment
source c2s-justin/bin/activate

# Run the training script
python train_lung_hce.py
```

## Reachability Matrix

The reachability matrix is a key component of HCE loss. It's a square matrix of size `(num_classes, num_classes)` where:
- Element `(i, j)` is 1 if class `j` is reachable from class `i` (i.e., `j` is `i` itself or a descendant of `i`)
- Element `(i, j)` is 0 otherwise

### Example:

For a hierarchy:
```
    Immune Cell
       ↙   ↓   ↘
  T cell B cell NK cell
```

The reachability matrix would be:
```
                Immune  T cell  B cell  NK cell
Immune Cell  |    1      1       1        1
T cell       |    0      1       0        0
B cell       |    0      0       1        0
NK cell      |    0      0       0        1
```

## Cell Type Ontology

For production use, you should use a proper cell ontology such as:
- [Cell Ontology (CL)](http://obofoundry.org/ontology/cl.html)
- CellxGene's cell type ontology
- Tissue-specific ontologies

The example in `train_lung_hce.py` uses a simplified keyword-based hierarchy for demonstration purposes.

## Environment Setup

Created virtual environment: `c2s-justin` (Python 3.13)

### Installation:

```bash
# Create virtual environment
python3 -m venv c2s-justin

# Activate
source c2s-justin/bin/activate

# Install Cell2Sentence in development mode
pip install -e .
```

## Configuration Options

The `HCETrainer` accepts the following parameters:

- `reachability_matrix` (torch.Tensor): The hierarchical reachability matrix
- `use_hce_loss` (bool): Whether to use HCE loss (True) or fall back to standard cross-entropy (False)
- All standard HuggingFace `Trainer` parameters

## Technical Details

### HCE Loss Computation:

1. Apply softmax to logits to get probabilities
2. Multiply probabilities by transpose of reachability matrix to propagate through hierarchy
3. Apply log transformation (with numerical stability term 1e-6)
4. Compute negative log-likelihood loss

### Mathematical Formula:

```
probs = softmax(logits)
probs_hierarchical = matmul(probs, reachability_matrix^T)
probs_hierarchical = log(probs_hierarchical + 1e-6)
loss = NLL(probs_hierarchical, labels)
```

## Compatibility

- Fully compatible with existing Cell2Sentence workflows
- Can be toggled on/off by setting `use_hce_loss=False`
- Works with any HuggingFace transformers model
- Supports distributed training, mixed precision, etc.

## Future Enhancements

Potential improvements:
1. Integration with Cell Ontology databases
2. Automatic hierarchy extraction from AnnData metadata
3. Support for multiple hierarchy levels with weighted contributions
4. Visualization tools for hierarchy and reachability matrices

## References

- **HCE Paper**: "Hierarchical cross-entropy loss improves atlas-scale single-cell annotation models" (bioRxiv: 10.1101/2025.04.23.650210)
- **Cell2Sentence Paper**: "Scaling Large Language Models for Next-Generation Single-Cell Analysis" (bioRxiv: 10.1101/2025.04.14.648850)

## Support

For questions or issues, please refer to:
- HCE GitHub: `/home/hugolab/Documents/HCE/HCE-HugoLab/hce-classification`
- Cell2Sentence GitHub: `/home/hugolab/Documents/Cell2Sentence-HugoLab/Cell2Sentence-HugoLab`
