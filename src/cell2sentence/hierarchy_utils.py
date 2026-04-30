"""
Hierarchy deduplication utilities for HCE + Cell2Sentence pipelines.

Problem: some datasets annotate cells with labels (e.g. "AT2", "Smooth muscle")
that also appear as interior nodes in the ontology — i.e. they are parents of more
specific subtypes.  HCE training propagates probability from subtypes to parents, so
the model learns these coarser names as intermediate targets and consistently predicts
the specific subtype instead of the generic parent, causing 0% recall for those classes.

Solution: rename the fine-level annotation of collision cells to "<label> (unspecified)"
and add that as a new leaf child of the original interior node in the ontology.
The model can then distinguish "AT2 (unspecified)" (cells that have no more specific
annotation) from the interior "AT2" node (the ancestor of AT2 proliferating, AT2 CCL3+,
etc.).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import pandas as pd


def deduplicate_hierarchy(
    obs: pd.DataFrame,
    ontology_dict: Dict[str, Optional[str]],
    ann_col: str,
    suffix: str = " (unspecified)",
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], pd.DataFrame]:
    """
    Rename finest-level annotations that collide with interior ontology nodes.

    A *collision* label is one that:
      1. Appears as a value in ``obs[ann_col]``  (it is used as a leaf annotation), AND
      2. Appears as a *parent* of at least one other node in ``ontology_dict``
         (it is also an interior node).

    For each collision label ``L``:
      - Every cell in ``obs`` where ``obs[ann_col] == L`` is re-annotated to
        ``L + suffix``  (default: ``"L (unspecified)"``).
      - A new entry ``{L + suffix: L}`` is added to ``ontology_dict`` so the
        renamed label becomes a proper leaf child of the original interior node.

    Args:
        obs:            ``adata.obs`` DataFrame (or a copy of it).
        ontology_dict:  Maps each cell type to its direct parent (``None`` for roots).
                        Modified in place.
        ann_col:        Column in ``obs`` that holds the finest-level annotation.
        suffix:         String appended to collision labels (default ``" (unspecified)"``).

    Returns:
        obs:            The (modified) obs DataFrame.
        ontology_dict:  The (modified) ontology dict with new leaf entries.
        report:         A DataFrame summarising every renamed label with columns
                        ``original_label``, ``new_label``, ``n_cells_renamed``.
    """
    # Interior nodes = labels that appear as a parent value in ontology_dict
    interior_nodes = {
        parent
        for parent in ontology_dict.values()
        if parent is not None
    }

    # Leaf annotations = unique values that appear in ann_col
    col = obs[ann_col].astype(str)
    leaf_annotations = set(col.dropna().unique())

    # Collisions = labels used as both interior nodes and finest-level annotations
    collisions = sorted(interior_nodes & leaf_annotations)

    if not collisions:
        report = pd.DataFrame(columns=["original_label", "new_label", "n_cells_renamed"])
        return obs, ontology_dict, report

    # Build a rename mapping and apply it in one vectorised pass over the
    # single annotation column — avoids copying the entire obs DataFrame.
    rename_map = {label: label + suffix for label in collisions}
    counts = col.value_counts()

    rows = []
    for label, new_label in rename_map.items():
        n = int(counts.get(label, 0))
        if n == 0:
            continue
        ontology_dict[new_label] = label   # new leaf → existing interior node
        rows.append({"original_label": label, "new_label": new_label, "n_cells_renamed": n})

    if rows:
        obs[ann_col] = col.replace(rename_map)

    report = pd.DataFrame(rows, columns=["original_label", "new_label", "n_cells_renamed"])
    return obs, ontology_dict, report


def find_collisions(
    obs: pd.DataFrame,
    ontology_dict: Dict[str, Optional[str]],
    ann_col: str,
) -> pd.DataFrame:
    """
    Identify collision labels without modifying anything.

    Useful for inspecting which labels will be affected before running
    ``deduplicate_hierarchy``.

    Returns a DataFrame with columns ``label``, ``n_cells``, ``n_children``
    (number of direct children in the ontology).
    """
    interior_nodes = {
        parent
        for parent in ontology_dict.values()
        if parent is not None
    }
    leaf_annotations = set(obs[ann_col].dropna().astype(str).unique())
    collisions = sorted(interior_nodes & leaf_annotations)

    # Count direct children for each collision label
    children_count: Dict[str, int] = {}
    for child, parent in ontology_dict.items():
        if parent in collisions:
            children_count[parent] = children_count.get(parent, 0) + 1

    label_counts = obs[ann_col].astype(str).value_counts()

    rows = []
    for label in collisions:
        rows.append({
            "label":      label,
            "n_cells":    int(label_counts.get(label, 0)),
            "n_children": children_count.get(label, 0),
        })

    return pd.DataFrame(rows, columns=["label", "n_cells", "n_children"])
