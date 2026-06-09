"""
Cell Ontology (CL) backbone utilities for HCE + Cell2Sentence pipelines.

The HCE reachability matrix needs a parent→child ontology.  Earlier multi-tissue
notebooks hand-curated this dict (``CROSS_ORGAN_HIERARCHY``), which drifted: casing
mismatches between curated nodes (``"Oligodendrocyte"``) and the lowercase Cell
Ontology terms coming from the data (``"oligodendrocyte"``) silently broke edges,
and several organs had no internal hierarchy at all.

This module builds a **hybrid** ontology instead:

  * a programmatic backbone derived from the Cell Ontology (CL) ``is_a`` DAG, and
  * curated overrides that attach atlas-specific labels (Siletti superclusters,
    activation/functional states, lung ``ann_finest_level`` names) to their nearest
    CL parent, and correct the handful of places where CL's ``is_a`` is biologically
    unhelpful for our purpose (e.g. CL places ``oligodendrocyte precursor cell``
    under ``neuron associated cell`` rather than alongside ``oligodendrocyte``).

The output is a flat ``{child: parent}`` dict (roots → ``None``) — the exact format
consumed by :func:`cell2sentence.hce_trainer.build_reachability_matrix_from_ontology`
and by :func:`cell2sentence.hierarchy_utils.deduplicate_hierarchy`.

``obonet`` is imported lazily so importing this module never requires it; only
:func:`load_cl_graph` does.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import warnings

# Cell Ontology root: CL:0000000 "cell".
CL_ROOT_ID = "CL:0000000"


# ======================================================================
# Loading the Cell Ontology
# ======================================================================


def load_cl_graph(obo_path: str):
    """
    Parse a Cell Ontology ``.obo`` file into a graph plus name lookups.

    Args:
        obo_path: Path to ``cl-basic.obo`` (or ``cl.obo``).

    Returns:
        graph:   The ``networkx.MultiDiGraph`` from ``obonet`` (edges are
                 child→parent, keyed by relation, e.g. ``"is_a"``).
        name2id: Lowercased CL term name → CL id (first occurrence wins).
        syn2id:  Lowercased CL *exact* synonym → CL id (only if not already a name).
        id2name: CL id → canonical (cased) term name.

    Only ``CL:`` nodes are indexed; imported GO/UBERON/PR terms are skipped.
    """
    import obonet  # lazy: keeps obonet an optional dependency

    graph = obonet.read_obo(obo_path)

    name2id: Dict[str, str] = {}
    syn2id: Dict[str, str] = {}
    id2name: Dict[str, str] = {}

    for nid, data in graph.nodes(data=True):
        if not str(nid).startswith("CL:"):
            continue
        name = data.get("name")
        if name:
            name = name.strip()
            id2name[nid] = name
            name2id.setdefault(name.lower(), nid)
        for syn in data.get("synonym", []):
            # obonet synonym format: '"text" SCOPE [refs]'
            parts = syn.split('"')
            if len(parts) < 3:
                continue
            text = parts[1].strip().lower()
            scope = parts[2].strip().split(" ", 1)[0].upper()
            if scope == "EXACT":
                syn2id.setdefault(text, nid)

    return graph, name2id, syn2id, id2name


def resolve_cl_id(
    label: str,
    name2id: Dict[str, str],
    syn2id: Dict[str, str],
    id_overrides: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Resolve a free-text label to a CL id via (1) explicit id override,
    (2) exact name match, (3) exact-synonym match.  Case-insensitive.

    Returns the CL id, or ``None`` if the label is not a CL term.
    """
    if id_overrides and label in id_overrides:
        return id_overrides[label]
    key = str(label).strip().lower()
    if key in name2id:
        return name2id[key]
    if key in syn2id:
        return syn2id[key]
    return None


def _is_a_parents(graph, nid: str) -> List[str]:
    """Direct ``is_a`` parents of a CL node (child→parent edges in obonet)."""
    return [
        v
        for _u, v, k in graph.out_edges(nid, keys=True)
        if k == "is_a" and str(v).startswith("CL:")
    ]


def _ancestor_chain_ids(
    graph,
    cid: str,
    id2name: Dict[str, str],
    stop_names: Set[str],
    max_depth: int,
) -> List[str]:
    """
    Walk ``is_a`` from ``cid`` toward the root, taking the first (primary) parent
    at each step.  Stops at the CL root, at a node whose name is in ``stop_names``,
    on a cycle, or after ``max_depth`` hops.

    CL is a DAG (some terms have multiple ``is_a`` parents); we keep the primary
    parent to fit the single-parent ``{child: parent}`` format the reachability
    builder expects.  Use ``curated_overrides`` to correct specific cases.
    """
    chain = [cid]
    seen = {cid}
    cur = cid
    while len(chain) <= max_depth:
        if id2name.get(cur, "").lower() in stop_names:
            break
        parents = _is_a_parents(graph, cur)
        if not parents:
            break
        nxt = parents[0]
        if nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        cur = nxt
    return chain


# ======================================================================
# Building the hybrid ontology
# ======================================================================


def build_cl_ontology(
    leaf_labels,
    graph,
    name2id: Dict[str, str],
    syn2id: Dict[str, str],
    id2name: Dict[str, str],
    curated_overrides: Optional[Dict[str, Optional[str]]] = None,
    id_overrides: Optional[Dict[str, str]] = None,
    roots: Optional[Set[str]] = None,
    max_depth: int = 100,
) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """
    Build a flat ``{child: parent}`` ontology from CL plus curated overrides.

    For every leaf label that resolves to a CL term, the full ``is_a`` chain to the
    root is added (the data label is kept verbatim as the leaf so it still matches
    the dataset annotations, while its ancestors use canonical CL names).  Labels
    that are not CL terms must be connected via ``curated_overrides``.

    Args:
        leaf_labels:       Iterable of (already canonicalized) cell-type labels.
        graph, name2id, syn2id, id2name: Output of :func:`load_cl_graph`.
        curated_overrides: ``{child: parent}`` applied *after* the CL backbone.
                           Use it to (a) attach non-CL atlas labels and
                           (b) override a CL-chosen parent (e.g. unify the
                           oligodendrocyte lineage).  ``parent`` may itself be a CL
                           term — its chain is added automatically.
        id_overrides:      ``{label: CL_id}`` to force resolution of ambiguous names.
        roots:             Names to treat as roots (chains stop there).  Matched
                           case-insensitively.  Defaults to empty (walk to CL root).
        max_depth:         Safety cap on chain length.

    Returns:
        ontology:    ``{child: parent}`` dict, roots mapped to ``None``.
        unresolved:  Labels that are neither CL terms nor covered by overrides.
    """
    curated_overrides = dict(curated_overrides or {})
    stop_names = {r.strip().lower() for r in (roots or set())}

    ontology: Dict[str, Optional[str]] = {}
    unresolved: List[str] = []

    def add_chain(leaf_label: str, cid: str) -> None:
        chain_ids = _ancestor_chain_ids(graph, cid, id2name, stop_names, max_depth)
        # First element keeps the verbatim data label; the rest use canonical names.
        names = [leaf_label] + [id2name.get(x, x) for x in chain_ids[1:]]
        for i in range(len(names) - 1):
            ontology.setdefault(names[i], names[i + 1])
        ontology.setdefault(names[-1], None)  # top of this chain is a root

    for label in leaf_labels:
        label = str(label)
        cid = resolve_cl_id(label, name2id, syn2id, id_overrides)
        if cid is None:
            unresolved.append(label)
            continue
        add_chain(label, cid)

    # Curated overrides: connect non-CL labels and apply structural corrections.
    for child, parent in curated_overrides.items():
        ontology[child] = parent  # may overwrite a CL-derived parent on purpose
        if parent is not None and parent not in ontology:
            pcid = resolve_cl_id(parent, name2id, syn2id, id_overrides)
            if pcid is not None:
                add_chain(parent, pcid)
            else:
                ontology.setdefault(parent, None)

    # Anything still unresolved after overrides is genuinely disconnected.
    still_unresolved = [l for l in unresolved if l not in ontology]

    # Ensure every referenced parent exists as a node.
    for parent in {p for p in ontology.values() if p is not None}:
        ontology.setdefault(parent, None)

    return ontology, still_unresolved


# ======================================================================
# Validation & traversal helpers
# ======================================================================


def get_ancestors(ontology: Dict[str, Optional[str]], node: str) -> List[str]:
    """Ordered list of ancestors of ``node`` (nearest first), excluding itself."""
    out: List[str] = []
    seen = {node}
    cur = ontology.get(node)
    while cur is not None and cur not in seen:
        out.append(cur)
        seen.add(cur)
        cur = ontology.get(cur)
    return out


def compute_depths(ontology: Dict[str, Optional[str]]) -> Dict[str, int]:
    """Depth of each node (roots = 0).  Robust to missing/cyclic entries."""
    depths: Dict[str, int] = {}

    def depth_of(n: str, stack: Set[str]) -> int:
        if n in depths:
            return depths[n]
        parent = ontology.get(n)
        if parent is None or n in stack:
            depths[n] = 0
            return 0
        d = depth_of(parent, stack | {n}) + 1
        depths[n] = d
        return d

    for node in ontology:
        depth_of(node, set())
    return depths


def validate_ontology(
    ontology: Dict[str, Optional[str]],
    leaf_labels=None,
    strict: bool = True,
) -> Dict[str, list]:
    """
    Check the ontology is well-formed for HCE.

    Detects: cycles, case-insensitive duplicate node names, leaves that do not
    reach a root, and (if ``leaf_labels`` given) leaves missing from the ontology.

    Args:
        ontology:    ``{child: parent}`` dict (roots → ``None``).
        leaf_labels: Optional iterable of expected leaf labels to check coverage.
        strict:      If True, raise ``ValueError`` on cycles or casing duplicates.

    Returns:
        A dict with keys ``cycles``, ``casing_dups``, ``unrooted``, ``missing_leaves``.
    """
    issues: Dict[str, list] = {
        "cycles": [],
        "casing_dups": [],
        "unrooted": [],
        "missing_leaves": [],
    }

    all_nodes: Set[str] = set(ontology.keys()) | {
        p for p in ontology.values() if p is not None
    }

    # Case-insensitive duplicate node names (the bug that broke v10 edges).
    by_lower: Dict[str, List[str]] = {}
    for node in all_nodes:
        by_lower.setdefault(node.lower(), []).append(node)
    issues["casing_dups"] = sorted(
        [sorted(v) for v in by_lower.values() if len(v) > 1]
    )

    # Cycle / reachability check: walk parents from each node.
    for node in ontology:
        seen: Set[str] = set()
        cur: Optional[str] = node
        reached_root = False
        while cur is not None:
            if cur in seen:
                issues["cycles"].append(node)
                break
            seen.add(cur)
            nxt = ontology.get(cur)
            if nxt is None:
                reached_root = True
                break
            cur = nxt
        if not reached_root and node not in issues["cycles"]:
            issues["unrooted"].append(node)

    if leaf_labels is not None:
        issues["missing_leaves"] = sorted(
            {str(l) for l in leaf_labels} - set(ontology.keys())
        )

    if strict and (issues["cycles"] or issues["casing_dups"]):
        raise ValueError(
            "Invalid ontology: "
            f"{len(issues['cycles'])} cycle(s), "
            f"{len(issues['casing_dups'])} casing-duplicate group(s). "
            f"Details: cycles={issues['cycles'][:5]}, "
            f"casing_dups={issues['casing_dups'][:5]}"
        )

    if issues["unrooted"]:
        warnings.warn(
            f"{len(issues['unrooted'])} node(s) do not reach a root "
            f"(first few: {issues['unrooted'][:5]})",
            stacklevel=2,
        )
    if issues["missing_leaves"]:
        warnings.warn(
            f"{len(issues['missing_leaves'])} leaf label(s) absent from ontology "
            f"(first few: {issues['missing_leaves'][:5]})",
            stacklevel=2,
        )

    return issues
