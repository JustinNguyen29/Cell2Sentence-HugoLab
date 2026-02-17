"""
Train Cell2Sentence with HCE Loss on lung.h5ad dataset

This script demonstrates how to:
1. Load the lung.h5ad dataset
2. Prepare it for Cell2Sentence training
3. Train with Hierarchical Cross-Entropy (HCE) loss
"""

import os
import sys
import scanpy as sc
import numpy as np
from transformers import TrainingArguments
from datasets import Dataset

# Add cell2sentence to path
sys.path.insert(0, '/home/hugolab/Documents/Cell2Sentence-HugoLab/Cell2Sentence-HugoLab/src')

from cell2sentence.csdata import CSData
from cell2sentence.csmodel import CSModel
from cell2sentence.hce_trainer import HCETrainer, build_reachability_matrix_from_ontology


def load_and_prepare_lung_data(h5ad_path):
    """Load and prepare the lung dataset."""
    print(f"Loading data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    print(f"Dataset shape: {adata.shape}")
    print(f"Available metadata: {adata.obs.columns.tolist()}")
    
    # Check what cell type annotations are available
    print("\nChecking for cell type columns...")
    cell_type_cols = [col for col in adata.obs.columns if 'cell' in col.lower() or 'type' in col.lower()]
    print(f"Potential cell type columns: {cell_type_cols}")
    
    return adata


def build_cell_type_hierarchy(adata, cell_type_column='cell_type'):
    """
    Build a simple cell type hierarchy from the data.
    
    For demonstration, we'll create a simple two-level hierarchy:
    - Broad cell types (e.g., immune, epithelial, stromal)
    - Specific cell types (e.g., T cell, B cell, etc.)
    """
    if cell_type_column not in adata.obs.columns:
        print(f"Warning: Column '{cell_type_column}' not found. Available columns:")
        print(adata.obs.columns.tolist())
        # Use first column that might be cell type
        if len(adata.obs.columns) > 0:
            cell_type_column = adata.obs.columns[0]
            print(f"Using column: {cell_type_column}")
    
    cell_types = adata.obs[cell_type_column].unique().tolist()
    print(f"\nFound {len(cell_types)} unique cell types")
    print(f"Cell types: {cell_types[:10]}...")  # Show first 10
    
    # Create a simple hierarchy - for demonstration purposes
    # In practice, you would use a proper cell ontology
    ontology_dict = {}
    
    # Example: group cell types by broad categories
    # This is a simplified example - you should use proper ontology
    immune_keywords = ['T cell', 'B cell', 'NK', 'macrophage', 'monocyte', 'dendritic']
    epithelial_keywords = ['epithelial', 'pneumocyte', 'ciliated', 'secretory']
    stromal_keywords = ['fibroblast', 'endothelial', 'smooth muscle', 'pericyte']
    
    for ct in cell_types:
        ct_lower = ct.lower()
        if any(keyword.lower() in ct_lower for keyword in immune_keywords):
            ontology_dict[ct] = 'Immune Cell'
        elif any(keyword.lower() in ct_lower for keyword in epithelial_keywords):
            ontology_dict[ct] = 'Epithelial Cell'
        elif any(keyword.lower() in ct_lower for keyword in stromal_keywords):
            ontology_dict[ct] = 'Stromal Cell'
        else:
            ontology_dict[ct] = 'Other Cell'
    
    # Add the broad categories to the list
    broad_types = ['Immune Cell', 'Epithelial Cell', 'Stromal Cell', 'Other Cell']
    all_cell_types = broad_types + cell_types
    
    print(f"\nCreated hierarchy with {len(all_cell_types)} total types (including broad categories)")
    
    return all_cell_types, ontology_dict, cell_type_column


def main():
    """Main training script."""
    
    # Configuration
    LUNG_H5AD_PATH = '/home/hugolab/Documents/Cell2Sentence-HugoLab/Cell2Sentence-HugoLab/lung.h5ad'
    SAVE_DIR = '/home/hugolab/Documents/Cell2Sentence-HugoLab/Cell2Sentence-HugoLab/outputs'
    MODEL_NAME = 'EleutherAI/pythia-160m'  # Small model for testing
    USE_HCE_LOSS = True
    TOP_K_GENES = 100
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load data
    adata = load_and_prepare_lung_data(LUNG_H5AD_PATH)
    
    # Build cell type hierarchy
    all_cell_types, ontology_dict, cell_type_column = build_cell_type_hierarchy(adata)
    
    # Build reachability matrix
    print("\nBuilding reachability matrix...")
    reachability_matrix = build_reachability_matrix_from_ontology(
        ontology_dict, 
        all_cell_types
    )
    print(f"Reachability matrix shape: {reachability_matrix.shape}")
    
    # Convert AnnData to Arrow dataset
    print("\nConverting AnnData to Arrow dataset...")
    csdata_obj = CSData.adata_to_arrow(
        adata,
        random_state=42,
        label_col_names=[cell_type_column]
    )
    print(f"CSData object created: {csdata_obj}")
    
    # Initialize C2S Model
    print(f"\nInitializing C2S model from {MODEL_NAME}...")
    csmodel = CSModel(
        model_name_or_path=MODEL_NAME,
        save_dir=SAVE_DIR,
        save_name='c2s_lung_hce'
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(SAVE_DIR, 'c2s_lung_hce_trained'),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(SAVE_DIR, 'logs'),
        logging_steps=10,
        save_steps=500,
        eval_steps=100,
        evaluation_strategy='steps',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='tensorboard',
        fp16=True,  # Use mixed precision if available
    )
    
    print("\nStarting training with HCE loss...")
    print(f"USE_HCE_LOSS: {USE_HCE_LOSS}")
    
    # Prepare trainer kwargs for HCE loss
    import torch
    trainer_kwargs = {
        'reachability_matrix': torch.tensor(reachability_matrix, dtype=torch.float32),
        'use_hce_loss': USE_HCE_LOSS
    }
    
    # Fine-tune with HCE loss
    csmodel.fine_tune(
        csdata=csdata_obj,
        task='cell_type_prediction',
        train_args=training_args,
        top_k_genes=TOP_K_GENES,
        max_eval_samples=500,
        trainer_class=HCETrainer,
        trainer_kwargs=trainer_kwargs
    )
    
    print("\nTraining completed!")
    print(f"Model saved to: {os.path.join(SAVE_DIR, 'c2s_lung_hce_trained')}")


if __name__ == '__main__':
    main()
