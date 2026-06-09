#!/usr/bin/env python
#
# Tests for the Cell Ontology (CL) backbone utilities.
#

from pathlib import Path

import networkx as nx
import pytest

from cell2sentence.ontology_utils import (
    build_cl_ontology,
    compute_depths,
    get_ancestors,
    load_cl_graph,
    resolve_cl_id,
    validate_ontology,
)
from cell2sentence.hce_trainer import build_reachability_matrix_from_ontology

HERE = Path(__file__).parent
# Repo-root download location used by the v11 pipeline (git-ignored).
# HERE = <repo>/src/cell2sentence/tests  → parents[2] = <repo>.
REAL_OBO = HERE.parents[2] / "ontology" / "cl-basic.obo"


def _synthetic_cl():
    """
    A tiny CL-like MultiDiGraph mirroring obonet's structure: nodes carry a
    ``name`` (+ optional ``synonym``); ``is_a`` edges point child→parent.

        cell
        ├── glial cell ── macroglial cell ── oligodendrocyte
        │                                  └─ astrocyte
        ├── neural cell ── neuron associated cell ── oligodendrocyte precursor cell
        └── leukocyte ── lymphocyte ── T cell
    """
    g = nx.MultiDiGraph()
    nodes = {
        "CL:0000000": "cell",
        "CL:0000125": "glial cell",
        "CL:0000126": "macroglial cell",
        "CL:0000128": "oligodendrocyte",
        "CL:0000127": "astrocyte",
        "CL:0002319": "neural cell",
        "CL:0000095": "neuron associated cell",
        "CL:0002453": "oligodendrocyte precursor cell",
        "CL:0000738": "leukocyte",
        "CL:0000542": "lymphocyte",
        "CL:0000084": "T cell",
    }
    for nid, name in nodes.items():
        g.add_node(nid, name=name)
    # An exact synonym for testing synonym resolution.
    g.nodes["CL:0002453"]["synonym"] = ['"OPC" EXACT [PMID:1]']

    isa = [
        ("CL:0000125", "CL:0000000"),  # glial cell -> cell
        ("CL:0000126", "CL:0000125"),  # macroglial -> glial
        ("CL:0000128", "CL:0000126"),  # oligodendrocyte -> macroglial
        ("CL:0000127", "CL:0000126"),  # astrocyte -> macroglial
        ("CL:0002319", "CL:0000000"),  # neural cell -> cell
        ("CL:0000095", "CL:0002319"),  # neuron associated cell -> neural cell
        ("CL:0002453", "CL:0000095"),  # OPC -> neuron associated cell  (CL's odd placement)
        ("CL:0000738", "CL:0000000"),  # leukocyte -> cell
        ("CL:0000542", "CL:0000738"),  # lymphocyte -> leukocyte
        ("CL:0000084", "CL:0000542"),  # T cell -> lymphocyte
    ]
    for child, parent in isa:
        g.add_edge(child, parent, key="is_a")

    name2id = {n.lower(): i for i, n in nodes.items()}
    syn2id = {"opc": "CL:0002453"}
    id2name = dict(nodes)
    return g, name2id, syn2id, id2name


class TestResolve:
    def test_name_and_synonym(self):
        g, name2id, syn2id, id2name = _synthetic_cl()
        assert resolve_cl_id("oligodendrocyte", name2id, syn2id) == "CL:0000128"
        assert resolve_cl_id("OLIGODENDROCYTE", name2id, syn2id) == "CL:0000128"
        assert resolve_cl_id("OPC", name2id, syn2id) == "CL:0002453"
        assert resolve_cl_id("not a cell type", name2id, syn2id) is None

    def test_id_override_wins(self):
        g, name2id, syn2id, id2name = _synthetic_cl()
        out = resolve_cl_id("oligodendrocyte", name2id, syn2id,
                            id_overrides={"oligodendrocyte": "CL:9999999"})
        assert out == "CL:9999999"


class TestBuildOntology:
    def test_chain_to_root(self):
        g, name2id, syn2id, id2name = _synthetic_cl()
        onto, unresolved = build_cl_ontology(
            ["oligodendrocyte", "T cell"], g, name2id, syn2id, id2name
        )
        assert unresolved == []
        assert onto["oligodendrocyte"] == "macroglial cell"
        assert onto["macroglial cell"] == "glial cell"
        assert onto["glial cell"] == "cell"
        assert onto["cell"] is None
        # ancestors walk all the way up
        assert get_ancestors(onto, "oligodendrocyte") == [
            "macroglial cell", "glial cell", "cell",
        ]

    def test_synonym_leaf_kept_verbatim(self):
        g, name2id, syn2id, id2name = _synthetic_cl()
        onto, unresolved = build_cl_ontology(["OPC"], g, name2id, syn2id, id2name)
        # the verbatim data label is preserved as the leaf...
        assert "OPC" in onto
        # ...and connected via the canonical chain.
        assert onto["OPC"] == "neuron associated cell"

    def test_roots_stop_chain(self):
        g, name2id, syn2id, id2name = _synthetic_cl()
        onto, _ = build_cl_ontology(
            ["oligodendrocyte"], g, name2id, syn2id, id2name,
            roots={"glial cell"},
        )
        assert onto["glial cell"] is None          # treated as a root
        assert "cell" not in onto                    # chain stopped before root

    def test_curated_override_unifies_oligo_lineage(self):
        # The biological correction we care about: put OPC alongside
        # oligodendrocyte under the glial lineage instead of CL's default.
        g, name2id, syn2id, id2name = _synthetic_cl()
        onto, unresolved = build_cl_ontology(
            ["oligodendrocyte", "oligodendrocyte precursor cell"],
            g, name2id, syn2id, id2name,
            curated_overrides={"oligodendrocyte precursor cell": "macroglial cell"},
        )
        assert unresolved == []
        assert onto["oligodendrocyte precursor cell"] == "macroglial cell"
        # both now share the macroglial/glial ancestry
        assert "glial cell" in get_ancestors(onto, "oligodendrocyte precursor cell")

    def test_curated_override_connects_non_cl_label(self):
        g, name2id, syn2id, id2name = _synthetic_cl()
        onto, unresolved = build_cl_ontology(
            ["macrophage_FOLR2"], g, name2id, syn2id, id2name,
            curated_overrides={"macrophage_FOLR2": "leukocyte"},
        )
        assert unresolved == []
        assert onto["macrophage_FOLR2"] == "leukocyte"
        assert onto["leukocyte"] == "cell"  # parent chain auto-added

    def test_genuinely_unresolved_reported(self):
        g, name2id, syn2id, id2name = _synthetic_cl()
        onto, unresolved = build_cl_ontology(
            ["totally unknown label"], g, name2id, syn2id, id2name
        )
        assert unresolved == ["totally unknown label"]


class TestValidate:
    def test_clean_ontology_passes(self):
        onto = {"oligodendrocyte": "glial cell", "glial cell": "cell", "cell": None}
        issues = validate_ontology(onto, leaf_labels=["oligodendrocyte"])
        assert issues["cycles"] == []
        assert issues["casing_dups"] == []
        assert issues["unrooted"] == []
        assert issues["missing_leaves"] == []

    def test_casing_duplicate_raises(self):
        onto = {"Oligodendrocyte": "glial cell", "oligodendrocyte": "glial cell",
                "glial cell": None}
        with pytest.raises(ValueError):
            validate_ontology(onto, strict=True)

    def test_cycle_raises(self):
        onto = {"a": "b", "b": "c", "c": "a"}
        with pytest.raises(ValueError):
            validate_ontology(onto, strict=True)

    def test_missing_leaf_warns(self):
        onto = {"cell": None}
        with pytest.warns(UserWarning):
            issues = validate_ontology(onto, leaf_labels=["oligodendrocyte"],
                                       strict=False)
        assert issues["missing_leaves"] == ["oligodendrocyte"]


class TestReachabilityIntegration:
    def test_feeds_reachability_builder(self):
        # The ontology must be consumable by the HCE reachability builder and
        # encode ancestor→descendant reachability correctly.
        g, name2id, syn2id, id2name = _synthetic_cl()
        onto, _ = build_cl_ontology(
            ["oligodendrocyte", "astrocyte", "T cell"], g, name2id, syn2id, id2name
        )
        classes = sorted(set(onto.keys()))
        R = build_reachability_matrix_from_ontology(onto, classes)
        idx = {c: i for i, c in enumerate(classes)}
        # glial cell reaches its descendant oligodendrocyte...
        assert R[idx["glial cell"], idx["oligodendrocyte"]] == 1
        # ...but a leaf does not reach its ancestor.
        assert R[idx["oligodendrocyte"], idx["glial cell"]] == 0
        # diagonal is always set
        assert R[idx["oligodendrocyte"], idx["oligodendrocyte"]] == 1

    def test_depths_monotonic(self):
        onto = {"oligodendrocyte": "glial cell", "glial cell": "cell", "cell": None}
        depths = compute_depths(onto)
        assert depths["cell"] == 0
        assert depths["glial cell"] == 1
        assert depths["oligodendrocyte"] == 2


@pytest.mark.skipif(not REAL_OBO.exists(), reason="cl-basic.obo not downloaded")
class TestRealCL:
    def test_load_and_build_real(self):
        graph, name2id, syn2id, id2name = load_cl_graph(str(REAL_OBO))
        assert len(name2id) > 2000
        leaves = ["oligodendrocyte", "astrocyte", "hepatocyte",
                  "regulatory T cell", "natural killer cell"]
        onto, unresolved = build_cl_ontology(
            leaves, graph, name2id, syn2id, id2name
        )
        assert unresolved == []
        # every requested leaf reaches the CL root "cell"
        for leaf in leaves:
            assert "cell" in get_ancestors(onto, leaf)
        validate_ontology(onto, leaf_labels=leaves, strict=True)
