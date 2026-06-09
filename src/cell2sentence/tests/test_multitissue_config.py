#!/usr/bin/env python
#
# Tests for the multi-tissue v11 experiment config (canonical vocab + hierarchy).
#
import sys
from pathlib import Path

import pytest

# multi_tissue_v11_config lives at the repo root alongside the notebooks.
REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))
REAL_OBO = REPO_ROOT / "ontology" / "cl-basic.obo"

cfg = pytest.importorskip("multi_tissue_v11_config")

from cell2sentence.ontology_utils import get_ancestors  # noqa: E402


class TestCanonicalize:
    def test_opc_variants_merge(self):
        # The three OPC spellings collapse to one canonical leaf.
        assert cfg.canonicalize("committed oligodendrocyte precursor") == \
            "oligodendrocyte precursor cell"
        assert cfg.canonicalize("oligodendrocyte_precursor_cell") == \
            "oligodendrocyte precursor cell"

    def test_lab_states_collapse_to_trainable(self):
        assert cfg.canonicalize("CD8_T_cell_late_exhausted") == \
            "CD8-positive, alpha-beta T cell"
        assert cfg.canonicalize("macrophage_FOLR2") == "macrophage"
        assert cfg.canonicalize("astrocyte_protoplasmic") == "astrocyte"

    def test_unknown_filter(self):
        assert cfg.is_valid_label("oligodendrocyte")
        for bad in ("Unknown", "none", "NA", "nan", ""):
            assert not cfg.is_valid_label(bad)

    def test_passthrough(self):
        assert cfg.canonicalize("hepatocyte") == "hepatocyte"


class TestAssignLineage:
    def test_opc_is_glial_not_progenitor(self):
        # Regression: "precursor cell" must not steal OPC into Hematopoietic
        # Progenitor before the Glial rule is checked.
        assert cfg.assign_lineage("oligodendrocyte precursor cell") == "Glial"
        assert cfg.assign_lineage("oligodendrocyte") == "Glial"

    def test_progenitors_still_route(self):
        assert cfg.assign_lineage("hematopoietic stem cell") == \
            "Hematopoietic Progenitor"
        assert cfg.assign_lineage("common myeloid progenitor") == "Myeloid"

    def test_choroid_plexus_stays_epithelial(self):
        assert cfg.assign_lineage("choroid plexus epithelial cell") == "Epithelial"

    def test_tumor_and_other(self):
        assert cfg.assign_lineage("AC-like") == "Tumor (glioma-like)"
        assert cfg.assign_lineage("some unmapped thing") == "Other"


@pytest.mark.skipif(not REAL_OBO.exists(), reason="cl-basic.obo not downloaded")
class TestBuildOntology:
    def _build(self):
        leaves = [
            "oligodendrocyte", "oligodendrocyte precursor cell", "astrocyte",
            "hepatocyte", "cholangiocyte", "regulatory T cell",
            "natural killer cell", "macrophage", "alveolar macrophage",
            "pericyte", "medium spiny neuron",
        ]
        # minimal lung subtree: leaf -> interior -> root, root anchored to CL
        lung_edges = {"AT2": "Alveolar epithelium",
                      "Alveolar epithelium": "Epithelial"}
        lung_roots = ["Epithelial"]
        return cfg.build_combined_ontology(
            leaves + ["AT2"], lung_edges, lung_roots, str(REAL_OBO), strict=True
        )

    def test_builds_clean_and_connected(self):
        onto, unresolved = self._build()
        assert unresolved == []
        # OPC and oligodendrocyte share the glial lineage (curated unification).
        assert "glial cell" in get_ancestors(onto, "oligodendrocyte")
        assert "glial cell" in get_ancestors(onto, "oligodendrocyte precursor cell")
        # medium spiny neuron routed to the neuron branch, not "secretory cell".
        assert "neuron" in get_ancestors(onto, "medium spiny neuron")
        # lung leaf joined the CL backbone via its anchored root.
        assert "epithelial cell" in get_ancestors(onto, "AT2")
