"""
Example script showing how to use HCE loss with Cell2Sentence
Simpler version for quick testing
"""

import os
import sys
import numpy as np
import torch
from transformers import TrainingArguments

# Add cell2sentence to path
sys.path.insert(0, '/home/hugolab/Documents/Cell2Sentence-HugoLab/Cell2Sentence-HugoLab/src')

from cell2sentence.csmodel import CSModel
from cell2sentence.hce_trainer import HCETrainer


def simple_hce_example():
    """
    Simplified example showing the key changes needed to use HCE loss.
    """
    
    # 1. Define your cell type hierarchy (example)
    # Format: child -> parent mapping
    ontology_dict = {
        "CD4 T cell": "T cell",
        "CD8 T cell": "T cell",
        "T cell": "Immune Cell",
        "B cell": "Immune Cell",
        "NK cell": "Immune Cell",
    }
    
    # All cell types including parent categories
    all_cell_types = ["Immune Cell", "T cell", "B cell", "NK cell", "CD4 T cell", "CD8 T cell"]
    
    # 2. Build reachability matrix
    from cell2sentence.hce_trainer import build_reachability_matrix_from_ontology
    
    reachability_matrix = build_reachability_matrix_from_ontology(
        ontology_dict, 
        all_cell_types
    )
    
    print("Reachability matrix shape:", reachability_matrix.shape)
    print("\nReachability matrix:")
    print(reachability_matrix)
    
    # 3. When calling fine_tune, add these parameters:
    
    # trainer_kwargs = {
    #     'reachability_matrix': torch.tensor(reachability_matrix, dtype=torch.float32),
    #     'use_hce_loss': True
    # }
    # 
    # csmodel.fine_tune(
    #     csdata=csdata_obj,
    #     task='cell_type_prediction',
    #     train_args=training_args,
    #     trainer_class=HCETrainer,  # <-- Use HCE trainer
    #     trainer_kwargs=trainer_kwargs  # <-- Pass HCE parameters
    # )
    
    print("\n" + "="*80)
    print("To use HCE loss, you need to:")
    print("1. Define your cell type hierarchy (ontology_dict)")
    print("2. Build the reachability matrix")
    print("3. Pass HCETrainer and trainer_kwargs to csmodel.fine_tune()")
    print("="*80)


if __name__ == '__main__':
    simple_hce_example()
