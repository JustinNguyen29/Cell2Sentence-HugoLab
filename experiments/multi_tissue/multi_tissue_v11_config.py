"""
Project-specific vocabulary + hierarchy configuration for the multi-tissue v11
HCE cell-type prediction experiments.

This is *experiment config*, not part of the reusable ``cell2sentence`` package —
the generic Cell-Ontology machinery lives in :mod:`cell2sentence.ontology_utils`.
Here we pin:

  * ``CANONICAL_LABEL_MAP`` — collapses near-duplicate / over-granular dataset
    labels onto a single canonical namespace shared by training and zero-shot eval
    (e.g. the three OPC variants → ``oligodendrocyte precursor cell``; lab
    activation states → their nearest trainable concept).  Applied to BOTH the
    training organ labels and the lab eval labels so they share one vocabulary.
  * ``CURATED_OVERRIDES`` — attaches non-CL atlas labels (Siletti neuron
    superclusters, HLCA lung roots) to their nearest Cell Ontology parent, and
    applies the handful of biological corrections where CL's primary ``is_a`` is
    unhelpful (oligodendrocyte-lineage unification; medium spiny neuron, which CL
    routes via "secretory cell" rather than the neuron branch).

``build_combined_ontology`` composes a fully-connected, validated ``{child:parent}``
ontology: CL backbone for CL-resolvable canonical leaves + curated overrides, then
the lung HLCA subtree grafted in (its roots anchored to CL), all rooted at CL "cell".

Verified against the real dataset vocabularies (285 canonical leaves): 0 cycles,
0 casing-duplicate nodes, 0 unrooted nodes, 0 missing leaves.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from cell2sentence.ontology_utils import (
    build_cl_ontology,
    load_cl_graph,
    resolve_cl_id,
    validate_ontology,
)

# HLCA lung uses its own annotation tree; these are its level-1 roots and the CL
# anchors we graft them onto so the lung subtree joins the unified CL hierarchy.
LUNG_HIERARCHY_COLS = [
    "ann_level_1", "ann_level_2", "ann_level_3", "ann_level_4", "ann_level_5",
]
LUNG_ROOT_ANCHORS = {
    "Immune": "hematopoietic cell",
    "Epithelial": "epithelial cell",
    "Endothelial": "endothelial cell",
    "Stroma": "connective tissue cell",
}

UNKNOWN_TOKENS = {"unknown", "none", "na", "n/a", "nan", ""}


def is_valid_label(value) -> bool:
    """False for Unknown/None/NA/NaN/empty annotations (see CLAUDE.md)."""
    s = str(value).strip().lower()
    return s not in UNKNOWN_TOKENS


def canonicalize(label: str) -> str:
    """Map a raw dataset label onto the shared canonical namespace."""
    return CANONICAL_LABEL_MAP.get(label, label)


def build_combined_ontology(
    canonical_leaves,
    lung_edges: Dict[str, str],
    lung_roots,
    obo_path: str,
    strict: bool = True,
) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """
    Compose the hybrid CL + lung + curated ontology.

    Args:
        canonical_leaves: Iterable of canonicalized leaf labels (training + eval),
                          Unknown already filtered.
        lung_edges:       ``{child: parent}`` HLCA lung edges, with leaf children
                          already canonicalized (interiors kept verbatim).
        lung_roots:       Lung level-1 root names to anchor onto CL.
        obo_path:         Path to ``cl-basic.obo``.
        strict:           Raise if the result has cycles or casing duplicates.

    Returns:
        ontology:   ``{child: parent}`` dict, roots → ``None`` (pre-dedup).
        unresolved: Canonical leaves that are neither CL terms, in the lung
                    subtree, nor covered by curated overrides (should be empty).
    """
    graph, name2id, syn2id, id2name = load_cl_graph(obo_path)

    leaves = list(dict.fromkeys(str(l) for l in canonical_leaves))
    cl_leaves = [l for l in leaves if resolve_cl_id(l, name2id, syn2id)]

    ontology, _ = build_cl_ontology(
        cl_leaves, graph, name2id, syn2id, id2name,
        curated_overrides=CURATED_OVERRIDES,
    )

    # Graft the lung subtree without overriding CL-derived edges.
    for child, parent in lung_edges.items():
        ontology.setdefault(child, parent)
    # Anchor lung roots onto CL (overwrite so the subtree joins the CL backbone).
    for r in lung_roots:
        if r in LUNG_ROOT_ANCHORS:
            ontology[r] = LUNG_ROOT_ANCHORS[r]

    # Ensure every referenced parent exists as a node.
    for parent in {p for p in ontology.values() if p is not None}:
        ontology.setdefault(parent, None)

    unresolved = [l for l in leaves if l not in ontology]
    validate_ontology(ontology, leaf_labels=leaves, strict=strict)
    return ontology, unresolved


# Coarse, keyword-based lineage assignment for the zero-shot lineage-match metric
# (independent of the ontology; robust to free-form atlas labels).  Ported from
# multi_tissue_hce_v10 with the Mph-suffix fix retained.
LINEAGE_RULES = [
    ("Myeloid", ["macroph", "monocyte", "microglia", "dendritic", " dc ", "tam-",
                 "tam_", "tam ", "kupffer", "neutrophil", "mast cell", "eosinophil",
                 "basophil", "myeloid", "granulocyte", "histiocyt", "langerhans",
                 " mph", "interstitial mph", "alveolar mph", "monocyte-derived",
                 "myelocyte", "promyelocyte", "mononuclear phagocyte"]),
    ("Lymphoid", ["t cell", "t_cell", "cd4", "cd8", "b cell", "b_cell", "nk cell",
                  "natural killer", "plasma cell", "plasmablast", "lymphocyte",
                  "lymphoid", "innate lymphoid", " ilc", "follicular helper",
                  "regulatory t", "gamma-delta", "germinal center"]),
    ("Erythroid/Megakaryocyte", ["erythro", "reticulocyt", "megakaryo", "platelet"]),
    # NB: Glial/Neuronal are checked BEFORE "Hematopoietic Progenitor" so that
    # "oligodendrocyte precursor cell" maps to Glial, not the generic
    # "precursor cell" → progenitor rule (a v10 bug that mis-lineaged all OPCs).
    ("Endothelial", ["endothel", "high endothelial venule", "ec arterial", "ec venous",
                     "ec general", "lymphatic ec", "arterial_endothel", "venous_endothel",
                     "capillary_endothel", "ec aerocyte", "ec capillary", "endothelial tip"]),
    ("Mesenchymal/Stromal", ["fibroblast", "stellate", "pericyte", "smooth muscle",
                             "smooth_muscle", "adipocyte", "mesenchym", "perivascular",
                             "meningeal", "stroma", "myofibro", "tenocyte", "chondrocyte"]),
    ("Epithelial", ["hepatocyt", "hepatoblast", "cholangio", "epithel", "enterocyt",
                    "alveolar type", "hepatic cell", "goblet", "keratino", "urothel", "tuft"]),
    ("Glial", ["astrocyt", "oligodendro", "opc", "glia", "bergmann", "ependymal",
               "choroid plexus"]),
    ("Neuronal", ["neuron", "interneuron", "intratelencephal", "hippocampal", "thalamic",
                  "amygdala", "rhombic", "mammillary", "medium spiny", " npc", "purkinje",
                  "granule cell", "msn", "gabaergic"]),
    ("Hematopoietic Progenitor", ["hematopoietic", "hsc", " progenitor", "precursor cell"]),
    ("Tumor (glioma-like)", ["ac-like", "mes-like", "npc-like", "opc-like"]),
]


def assign_lineage(label) -> str:
    """Coarse lineage of a free-form label, or ``"Other"``."""
    if label is None:
        return "Other"
    s = " " + str(label).lower().strip() + " "
    for lineage, kws in LINEAGE_RULES:
        for kw in kws:
            if kw in s:
                return lineage
    return "Other"


CANONICAL_LABEL_MAP: Dict[str, str] = {
    'Alveolar macrophages': 'alveolar macrophage',
    'B cells': 'B cell',
    'B_cell': 'B cell',
    'B_cell_naive': 'naive B cell',
    'CD14-positive monocyte': 'classical monocyte',
    'CD14-positive, CD16-positive monocyte': 'intermediate monocyte',
    'CD4 T cells': 'CD4-positive, alpha-beta T cell',
    'CD4-positive helper T cell': 'CD4-positive, alpha-beta T cell',
    'CD4-positive, CD25-positive, alpha-beta regulatory T cell': 'regulatory T cell',
    'CD4_T_cell_activated': 'CD4-positive, alpha-beta T cell',
    'CD4_T_cell_naive_or_memory': 'CD4-positive, alpha-beta T cell',
    'CD8 T cells': 'CD8-positive, alpha-beta T cell',
    'CD8-positive, alpha-beta memory T cell, CD45RO-positive': 'CD8-positive, alpha-beta memory T cell',
    'CD8_T_cell_early_activated': 'CD8-positive, alpha-beta T cell',
    'CD8_T_cell_late_exhausted': 'CD8-positive, alpha-beta T cell',
    'CD8_T_cell_proliferating': 'CD8-positive, alpha-beta T cell',
    'Classical monocytes': 'classical monocyte',
    'Eccentric medium spiny neuron': 'eccentric medium spiny neuron',
    'GABAergic_interneuron': 'GABAergic interneuron',
    'GABAergic_interneuron_SST': 'GABAergic interneuron',
    'LN_fibroblast_collagen_high': 'fibroblast',
    'LN_stroma_cell': 'stromal cell',
    'LN_stroma_cell_CHI3L1': 'stromal cell',
    'Mast cells': 'mast cell',
    'Medium spiny neuron': 'medium spiny neuron',
    'NK cells': 'natural killer cell',
    'NK_cell': 'natural killer cell',
    'Non-classical monocytes': 'non-classical monocyte',
    'Plasma cells': 'plasma cell',
    'Plasmacytoid DCs': 'plasmacytoid dendritic cell',
    'Smooth muscle': 'smooth muscle cell',
    'Treg_cell': 'regulatory T cell',
    'alveolar_macrophage': 'alveolar macrophage',
    'arterial_endothelial_cell': 'endothelial cell of artery',
    'astrocyte_fibrous_like': 'astrocyte',
    'astrocyte_protoplasmic': 'astrocyte',
    'cDC1': 'conventional dendritic cell',
    'capillary_endothelial_cell': 'capillary endothelial cell',
    'committed oligodendrocyte precursor': 'oligodendrocyte precursor cell',
    'cycling plasma cell': 'plasma cell',
    'dendritic cell, human': 'dendritic cell',
    'effector memory CD4-positive, alpha-beta T cell, terminally differentiated': 'effector memory CD4-positive, alpha-beta T cell',
    'endothelial cell of pericentral hepatic sinusoid': 'endothelial cell of hepatic sinusoid',
    'endothelial cell of periportal hepatic sinusoid': 'endothelial cell of hepatic sinusoid',
    'endothelial_cell_proliferating': 'endothelial cell',
    'follicular_dendritic_cell': 'follicular dendritic cell',
    'granulocyte monocyte progenitor cell': 'granulocyte monocyte progenitor cell',
    'group 3 innate lymphoid cell, human': 'group 3 innate lymphoid cell',
    'hepatic_stellate_cell_ACTA2_high': 'hepatic stellate cell',
    'hepatic_stellate_cell_ACTA2_high_collagen_high': 'hepatic stellate cell',
    'hepatic_stellate_cell_HGF_high': 'hepatic stellate cell',
    'inflammatory macrophage': 'macrophage',
    'intrahepatic cholangiocyte': 'cholangiocyte',
    'liver dendritic cell': 'conventional dendritic cell',
    'liver_pericentral_capillary_endothelial_cell': 'endothelial cell of hepatic sinusoid',
    'liver_periportal_capillary_endothelial_cell': 'endothelial cell of hepatic sinusoid',
    'lymphatic_endothelial_cell': 'endothelial cell of lymphatic vessel',
    'macrophage_APOE_CHIT': 'macrophage',
    'macrophage_C3': 'macrophage',
    'macrophage_F13A1': 'macrophage',
    'macrophage_FOLR2': 'macrophage',
    'macrophage_ISG_expressing': 'macrophage',
    'macrophage_VEGFA': 'macrophage',
    'macrophage_glycolytic': 'macrophage',
    'mast_cell': 'mast cell',
    'mature B cell': 'B cell',
    'mature NK T cell': 'mature NK T cell',
    'mature alpha-beta T cell': 'alpha-beta T cell',
    'meningeal_fibroblast': 'fibroblast',
    'monocyte_AREG_EREG': 'monocyte',
    'monocyte_CSF3R': 'classical monocyte',
    'monocyte_ITGAL': 'non-classical monocyte',
    'monocyte_SOCS3': 'classical monocyte',
    'myeloid dendritic cell': 'conventional dendritic cell',
    'oligodendrocyte_ISG_expressing': 'oligodendrocyte',
    'oligodendrocyte_precursor_cell': 'oligodendrocyte precursor cell',
    'perivascular_fibroblast': 'fibroblast',
    'plasma_cell': 'plasma cell',
    'plasma_cell_proliferating': 'plasma cell',
    'plasmacytoid dendritic cell, human': 'plasmacytoid dendritic cell',
    'smooth_muscle_cell': 'smooth muscle cell',
    'vein endothelial cell': 'endothelial cell of vein',
    'venous_endothelial_cell': 'endothelial cell of vein',
}

CURATED_OVERRIDES: Dict[str, Optional[str]] = {
    'Amygdala excitatory': 'excitatory neuron',
    'CGE interneuron': 'inhibitory neuron',
    'Cerebellar inhibitory': 'inhibitory neuron',
    'Deep-layer corticothalamic and 6b': 'excitatory neuron',
    'Deep-layer intratelencephalic': 'excitatory neuron',
    'Deep-layer near-projecting': 'excitatory neuron',
    'Endothelial': 'endothelial cell',
    'Epithelial': 'epithelial cell',
    'Hippocampal CA1-3': 'excitatory neuron',
    'Hippocampal CA4': 'excitatory neuron',
    'Hippocampal dentate gyrus': 'excitatory neuron',
    'Immune': 'hematopoietic cell',
    'LAMP5-LHX6 and Chandelier': 'inhibitory neuron',
    'Lower rhombic lip': 'excitatory neuron',
    'MGE interneuron': 'inhibitory neuron',
    'Mammillary body': 'excitatory neuron',
    'Midbrain-derived inhibitory': 'inhibitory neuron',
    'Miscellaneous': 'neuron',
    'Stroma': 'connective tissue cell',
    'Thalamic excitatory': 'excitatory neuron',
    'Upper rhombic lip': 'excitatory neuron',
    'Upper-layer intratelencephalic': 'excitatory neuron',
    'excitatory neuron': 'neuron',
    'inhibitory neuron': 'neuron',
    'medium spiny neuron': 'inhibitory neuron',
    'oligodendrocyte precursor cell': 'macroglial cell',
}
