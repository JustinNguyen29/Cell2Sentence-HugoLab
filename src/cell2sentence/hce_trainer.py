"""
Custom Trainer with Hierarchical Cross-Entropy Loss for Cell2Sentence

Implements HCE loss from:
  "Improving atlas-scale single-cell annotation models with hierarchical cross-entropy loss"
  Nature Computational Science, 2026. https://doi.org/10.1038/s43588-025-00945-z

For a causal language model (Cell2Sentence), HCE is applied at the token positions
where the label is the first vocabulary token of a cell type name.  At those positions,
the model's full vocabulary distribution is restricted to the set of cell type first-tokens,
the reachability matrix propagates probability mass from descendants to ancestors, and the
weighted negative log-likelihood (Eq. 7 in the paper) is computed.  All other positions use
standard causal-LM cross-entropy.
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, List, Optional

import numpy as np


class HCETrainer(Trainer):
    """
    Custom HuggingFace Trainer that replaces standard cross-entropy with
    Hierarchical Cross-Entropy (HCE) loss at cell type prediction positions.

    The reachability matrix R (C × C) encodes the cell ontology DAG:
        R[i, j] = 1  if cell type j is reachable from cell type i
                      (j == i, or j is a descendant of i in the ontology)
        R[i, j] = 0  otherwise

    Adjusted scores are computed as s = p @ R^T, so that
        s[i] = p[i] + Σ_{j ∈ descendants(i)} p[j]
    which accumulates probability mass from fine-grained subtypes into their
    ancestors, enforcing the consistency constraint from the paper.

    The per-sample HCE loss for a sample with true cell type t is (paper Eq. 7):
        L_HCE(x) = -w_t * log(s_t + ε),   ε = 1e-8
    where  w_i = N / (C * n_i)  are the balanced class weights (scikit-learn convention).

    Args:
        reachability_matrix: Float tensor of shape (C, C) encoding the ontology DAG.
            R[i, j] = 1 if j is reachable from i (parent→child semantics).
        cell_type_token_ids: List of length C with the first vocabulary token ID for
            each cell type name (same order as rows/columns of reachability_matrix).
            Used to identify cell type prediction positions in the token sequence.
        class_weights: Float tensor of shape (C,) with per-class weights
            w_i = N / (C * n_i). Pass None to use uniform weights.
        use_hce_loss: Set False to disable HCE and fall back to standard causal-LM CE.
        **kwargs: Forwarded to the base HuggingFace Trainer.
    """

    def __init__(
        self,
        reachability_matrix: Optional[torch.Tensor] = None,
        cell_type_token_ids: Optional[List[int]] = None,
        class_weights: Optional[torch.Tensor] = None,
        use_hce_loss: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_hce_loss = use_hce_loss
        self.cell_type_token_ids = cell_type_token_ids
        self.class_weights = class_weights
        self.reachability_matrix = None

        if self.use_hce_loss and reachability_matrix is not None:
            if not isinstance(reachability_matrix, torch.Tensor):
                reachability_matrix = torch.tensor(
                    reachability_matrix, dtype=torch.float32
                )
            self.reachability_matrix = reachability_matrix

    # ------------------------------------------------------------------
    # Core loss
    # ------------------------------------------------------------------

    def hierarchical_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the HCE loss for a causal language model batch.

        For token positions where the (shifted) label is the first token of a
        cell type name, HCE replaces the standard CE loss:
            1. Restrict logits to the C cell-type first-token columns.
            2. Softmax over that subspace → probabilities p  (n_ct × C)
            3. Adjusted scores: s = p @ R^T  (paper Eq. 4)
            4. Weighted loss: -w_t * log(s_t + 1e-8)  (paper Eq. 7)

        For all other active positions, standard causal-LM cross-entropy is used.
        The final loss is the mean over all active (non-masked) token positions.

        Args:
            logits: Raw model output, shape (batch, seq_len, vocab_size).
            labels: Token IDs with -100 for ignored positions,
                    shape (batch, seq_len). NOT pre-shifted — the shift is
                    applied here to match HuggingFace's CausalLM convention.

        Returns:
            Scalar mean loss.
        """
        device = logits.device

        # --- Causal-LM shift -------------------------------------------
        # logits[t] is the distribution over the NEXT token (position t+1).
        # So we compare logits[0..T-2] against labels[1..T-1].
        shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[..., 1:].contiguous()        # (B, T-1)

        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)    # (N, V)
        flat_labels = shift_labels.view(-1)                # (N,)

        # Keep only positions that contribute to the loss
        active_mask = flat_labels != -100
        if not active_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        active_logits = flat_logits[active_mask]           # (N_act, V)
        active_labels = flat_labels[active_mask]           # (N_act,)
        n_active = active_logits.shape[0]

        # --- Identify cell-type prediction positions -------------------
        # A position is a cell-type position if its target token is the
        # first token of a cell type name.  We build a lookup table from
        # token_id → cell_type_index.  Tokens shared by multiple cell type
        # names are marked as ambiguous (-2) and fall back to standard CE.
        can_apply_hce = (
            self.reachability_matrix is not None
            and self.cell_type_token_ids is not None
        )

        ct_mask = torch.zeros(n_active, dtype=torch.bool, device=device)
        ct_true_idx = torch.full((n_active,), -1, dtype=torch.long, device=device)

        if can_apply_hce:
            ct_lookup = torch.full(
                (vocab_size,), -1, dtype=torch.long, device=device
            )
            seen: Dict[int, int] = {}
            for ct_idx, tok_id in enumerate(self.cell_type_token_ids):
                if tok_id in seen:
                    ct_lookup[tok_id] = -2  # ambiguous — fall back to CE
                else:
                    ct_lookup[tok_id] = ct_idx
                    seen[tok_id] = ct_idx

            active_ct = ct_lookup[active_labels]           # (N_act,)
            ct_mask = active_ct >= 0
            ct_true_idx[ct_mask] = active_ct[ct_mask]

        # --- Standard CE for non-cell-type positions -------------------
        non_ct_mask = ~ct_mask
        ce_sum = torch.tensor(0.0, device=device)
        if non_ct_mask.any():
            ce_sum = F.cross_entropy(
                active_logits[non_ct_mask],
                active_labels[non_ct_mask],
                reduction="sum",
            )

        # --- HCE for cell-type prediction positions --------------------
        hce_sum = torch.tensor(0.0, device=device)
        if ct_mask.any() and can_apply_hce:
            R = self.reachability_matrix.to(device)        # (C, C)
            ct_tok_t = torch.tensor(
                self.cell_type_token_ids, dtype=torch.long, device=device
            )                                              # (C,)

            ct_logits = active_logits[ct_mask]             # (N_ct, V)
            ct_labels = ct_true_idx[ct_mask]               # (N_ct,)

            # Restrict to the C cell-type first-token columns
            ct_vocab_logits = ct_logits[:, ct_tok_t]       # (N_ct, C)

            # Softmax over the cell-type vocabulary subspace
            ct_probs = F.softmax(ct_vocab_logits, dim=-1)  # (N_ct, C)

            # Propagate probability mass upward (paper Eq. 4):
            #   s[b, i] = Σ_j p[b, j] * R[i, j]  →  S = P @ R^T
            adjusted = torch.matmul(ct_probs, R.T)         # (N_ct, C)

            # Log with ε = 1e-8 for numerical stability (paper Eq. 7)
            log_adjusted = torch.log(adjusted + 1e-8)      # (N_ct, C)

            # Weighted NLL: L_HCE(x) = -w_t * log(s_t + ε)
            n_ct = ct_mask.sum()
            arange_ct = torch.arange(n_ct, device=device)
            log_st = log_adjusted[arange_ct, ct_labels]    # (N_ct,)

            if self.class_weights is not None:
                weights = self.class_weights.to(device)    # (C,)
                hce_sum = -(weights[ct_labels] * log_st).sum()
            else:
                hce_sum = -log_st.sum()

        # Mean over all active positions (CE + HCE combined)
        return (ce_sum + hce_sum) / n_active

    # ------------------------------------------------------------------
    # Trainer hook
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override Trainer.compute_loss to inject HCE."""
        # Pop labels so the model does NOT compute its own internal CE loss.
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)

        if labels is not None:
            logits = outputs.get("logits")

            if self.use_hce_loss and self.reachability_matrix is not None:
                loss = self.hierarchical_cross_entropy_loss(logits, labels)
            else:
                # Standard causal-LM CE with the required shift
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            outputs["loss"] = loss
        else:
            loss = outputs.get("loss")

        return (loss, outputs) if return_outputs else loss


# ======================================================================
# Reachability-matrix builders
# ======================================================================


def build_reachability_matrix_from_hierarchy(
    cell_type_hierarchy: Dict[str, any],
    cell_type_to_idx: Dict[str, int],
) -> np.ndarray:
    """
    Build a reachability matrix from a nested hierarchy dictionary.

    Args:
        cell_type_hierarchy: Nested dict, e.g.
            {"Lymphocyte": {"children": ["T cell", "B cell"]},
             "T cell":     {"children": ["CD4+ T cell", "CD8+ T cell"]}}
        cell_type_to_idx: Mapping from cell type name → row/column index.

    Returns:
        Float32 array of shape (C, C).  R[i, j] = 1 if j is reachable from i
        (i.e. j == i or j is a descendant of i).
    """
    num_classes = len(cell_type_to_idx)
    reachability = np.eye(num_classes, dtype=np.float32)

    def _add_descendants(parent_name: str, ancestor_idx: int) -> None:
        if parent_name not in cell_type_hierarchy:
            return
        for child_name in cell_type_hierarchy[parent_name].get("children", []):
            if child_name in cell_type_to_idx:
                child_idx = cell_type_to_idx[child_name]
                reachability[ancestor_idx, child_idx] = 1
                _add_descendants(child_name, ancestor_idx)  # transitive

    for cell_type, idx in cell_type_to_idx.items():
        _add_descendants(cell_type, idx)

    return reachability


def build_reachability_matrix_from_ontology(
    ontology_dict: Dict[str, str],
    cell_types: List[str],
) -> np.ndarray:
    """
    Build a reachability matrix from a flat parent-child ontology dict.

    Args:
        ontology_dict: Maps each cell type to its direct parent, e.g.
            {"T cell": "Lymphocyte", "B cell": "Lymphocyte"}.
            Root nodes map to None.
        cell_types: Ordered list of all cell types (defines matrix index order).

    Returns:
        Float32 array of shape (C, C).  R[i, j] = 1 if j is reachable from i
        (i.e. j == i, or j is a descendant of i in the ontology DAG).
    """
    num_classes = len(cell_types)
    cell_type_to_idx = {ct: i for i, ct in enumerate(cell_types)}

    reachability = np.eye(num_classes, dtype=np.float32)

    # Direct parent → child edges
    for child, parent in ontology_dict.items():
        if parent is None:
            continue
        if child in cell_type_to_idx and parent in cell_type_to_idx:
            reachability[cell_type_to_idx[parent], cell_type_to_idx[child]] = 1

    # Transitive closure (Floyd-Warshall)
    for k in range(num_classes):
        for i in range(num_classes):
            if reachability[i, k] == 0:
                continue
            for j in range(num_classes):
                if reachability[k, j]:
                    reachability[i, j] = 1

    return reachability


# ======================================================================
# Utility helpers
# ======================================================================


def get_cell_type_first_token_ids(
    cell_type_names: List[str],
    tokenizer,
) -> List[int]:
    """
    Return the first vocabulary token ID for each cell type name.

    These IDs identify cell type prediction positions in the training sequence
    and restrict the vocabulary during HCE loss computation.

    Args:
        cell_type_names: Ordered list of cell type names (same order as the
            rows/columns of the reachability matrix).
        tokenizer: HuggingFace tokenizer for the model being trained.

    Returns:
        List of length C with one token ID per cell type.
    """
    first_token_ids = []
    for name in cell_type_names:
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        if token_ids:
            first_token_ids.append(token_ids[0])
        else:
            first_token_ids.append(tokenizer.unk_token_id)
    return first_token_ids


def compute_hce_class_weights(
    cell_types: List[str],
    cell_type_counts: Dict[str, int],
) -> np.ndarray:
    """
    Compute balanced class weights following scikit-learn's convention.

    Per the paper (Eq. 7):  w_i = N / (C * n_i)
    where N = total training samples, C = number of observed classes,
    n_i = number of training samples for class i.

    Cell types present in the hierarchy but absent from the training data
    receive weight 0.

    Args:
        cell_types: Ordered list of all cell types (same order as reachability matrix).
        cell_type_counts: Dict mapping cell type name → count of training samples.

    Returns:
        Float32 array of shape (C,) with one weight per cell type.
    """
    observed = {ct: cnt for ct, cnt in cell_type_counts.items() if cnt > 0}
    N = sum(observed.values())
    C = len(observed)

    weights = np.zeros(len(cell_types), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        if ct in observed:
            weights[i] = N / (C * observed[ct])

    return weights
