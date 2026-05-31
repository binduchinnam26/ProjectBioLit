"""
layout_engine.py — Two-stage UMAP + ForceAtlas2 layout pipeline.

Replicates VOSviewer's organic, cluster-separated VOS layout:
  Stage 1: UMAP semantic positioning from paper embeddings
  Stage 2: ForceAtlas2 refinement using graph edge weights as attraction
  Post:     Cluster separation enhancement to push islands apart

Falls back to kamada_kawai / spring_layout when embeddings are unavailable
or when optional dependencies (umap-learn, fa2) are not installed.

Output positions are always normalised to approximately [-1, 1] so the
existing rendering code (_LAYOUT_SCALE multiplications) works unchanged.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

import config

logger = logging.getLogger(__name__)


# ── Per-node embedding computation ───────────────────────────────────────────

def _avg(indices: List[int], arr: np.ndarray) -> Optional[np.ndarray]:
    valid = [i for i in indices if 0 <= i < len(arr)]
    return arr[valid].mean(axis=0) if valid else None


def _node_embeddings_coauth(
    G: nx.Graph,
    papers_df,
    embeddings_array: np.ndarray,
) -> Dict[str, np.ndarray]:
    author_idx: Dict[str, List[int]] = {}
    for idx, row in enumerate(papers_df.to_dict("records")):
        for a in (row.get("authors") or []):
            name = a["name"] if isinstance(a, dict) else str(a)
            if name:
                author_idx.setdefault(name, []).append(idx)
    result = {}
    for node in G.nodes():
        emb = _avg(author_idx.get(str(node), []), embeddings_array)
        if emb is not None:
            result[str(node)] = emb
    return result


def _node_embeddings_keyword(
    G: nx.Graph,
    papers_df,
    embeddings_array: np.ndarray,
) -> Dict[str, np.ndarray]:
    kw_idx: Dict[str, List[int]] = {}
    for idx, row in enumerate(papers_df.to_dict("records")):
        for kw in (row.get("author_keywords") or []):
            if kw:
                kw_idx.setdefault(kw, []).append(idx)
        for mesh in (row.get("mesh_terms") or []):
            if isinstance(mesh, dict):
                for field in ("descriptor", "qualifier"):
                    val = mesh.get(field, "")
                    if val:
                        kw_idx.setdefault(val, []).append(idx)
        for chem in (row.get("chemicals") or []):
            name = chem.get("name", "") if isinstance(chem, dict) else (chem if isinstance(chem, str) else "")
            if name:
                kw_idx.setdefault(name, []).append(idx)
    result = {}
    for node in G.nodes():
        emb = _avg(kw_idx.get(str(node), []), embeddings_array)
        if emb is not None:
            result[str(node)] = emb
    return result


def _node_embeddings_topic(
    G: nx.Graph,
    papers_df,
    embeddings_array: np.ndarray,
    doc_topics_df,
) -> Dict[str, np.ndarray]:
    if doc_topics_df is None or doc_topics_df.empty:
        return {}
    pmid_to_idx = {str(r["pmid"]): i for i, r in enumerate(papers_df.to_dict("records"))}
    topic_idx: Dict[int, List[int]] = {}
    for _, row in doc_topics_df.iterrows():
        tid = int(row.get("topic_id", -1))
        if tid < 0:
            continue
        idx = pmid_to_idx.get(str(row.get("pmid", "")))
        if idx is not None:
            topic_idx.setdefault(tid, []).append(idx)
    result = {}
    for node in G.nodes():
        try:
            nid = int(node)
        except (ValueError, TypeError):
            continue
        emb = _avg(topic_idx.get(nid, []), embeddings_array)
        if emb is not None:
            result[str(node)] = emb
    return result


def _node_embeddings_kg(
    G: nx.Graph,
    papers_df,
    embeddings_array: np.ndarray,
    entities_df,
) -> Dict[str, np.ndarray]:
    if entities_df is None or entities_df.empty:
        return {}
    pmid_to_idx = {str(r["pmid"]): i for i, r in enumerate(papers_df.to_dict("records"))}
    text_col = next((c for c in entities_df.columns if c in ("entity_text", "text", "name")), None)
    pmid_col = next((c for c in entities_df.columns if "pmid" in c.lower()), None)
    if not text_col or not pmid_col:
        return {}
    ent_idx: Dict[str, List[int]] = {}
    for _, row in entities_df.iterrows():
        ent = str(row.get(text_col, ""))
        idx = pmid_to_idx.get(str(row.get(pmid_col, "")))
        if ent and idx is not None:
            ent_idx.setdefault(ent, []).append(idx)
    result = {}
    for node in G.nodes():
        emb = _avg(ent_idx.get(str(node), []), embeddings_array)
        if emb is not None:
            result[str(node)] = emb
    return result


# ── Stage 1: UMAP ─────────────────────────────────────────────────────────────

def _run_umap(node_embeddings: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    try:
        import umap as umap_module
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP stage")
        return {}

    nodes = list(node_embeddings.keys())
    n = len(nodes)
    if n < 4:
        return {}

    matrix = np.array([node_embeddings[nd] for nd in nodes], dtype=np.float32)
    n_neighbors = min(15, n - 1)

    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.3,
        metric="cosine",
        random_state=42,
        n_jobs=1,
        spread=3.0,
        low_memory=False,
    )
    try:
        pos2d = reducer.fit_transform(matrix)
    except Exception as exc:
        logger.warning("UMAP failed: %s", exc)
        return {}

    min_x, max_x = pos2d[:, 0].min(), pos2d[:, 0].max()
    min_y, max_y = pos2d[:, 1].min(), pos2d[:, 1].max()
    xr = max_x - min_x + 1e-9
    yr = max_y - min_y + 1e-9

    return {
        nodes[i]: (
            float((pos2d[i, 0] - min_x) / xr * 1800 + 100),
            float((pos2d[i, 1] - min_y) / yr * 900 + 100),
        )
        for i in range(n)
    }


# ── Stage 2: ForceAtlas2 ──────────────────────────────────────────────────────

def _run_forceatlas2(
    G: nx.Graph,
    umap_pos: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    try:
        from fa2 import ForceAtlas2
    except ImportError:
        logger.debug("fa2 not installed; skipping ForceAtlas2 stage")
        return umap_pos

    # FA2 works on undirected simple graphs
    G_simple = nx.Graph(G.to_undirected() if G.is_directed() else G)
    nodes_with_pos = [n for n in G_simple.nodes() if str(n) in umap_pos]
    if len(nodes_with_pos) < 2:
        return umap_pos

    G_sub = G_simple.subgraph(nodes_with_pos).copy()
    pos_for_fa2 = {n: list(umap_pos[str(n)]) for n in G_sub.nodes()}

    fa2 = ForceAtlas2(
        outboundAttractionDistribution=True,
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.5,
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=0.05,
        verbose=False,
    )
    try:
        raw = fa2.forceatlas2_networkx_layout(G_sub, pos=pos_for_fa2, iterations=500)
        result = {str(n): (float(x), float(y)) for n, (x, y) in raw.items()}
        # Pass through any nodes not in the subgraph
        for ns, xy in umap_pos.items():
            if ns not in result:
                result[ns] = xy
        return result
    except Exception as exc:
        logger.warning("ForceAtlas2 failed: %s", exc)
        return umap_pos


# ── Post-process: cluster separation ─────────────────────────────────────────

def _enhance_cluster_separation(
    positions: Dict[str, Tuple[float, float]],
    node_communities: Dict[str, int],
    separation_factor: float = 1.4,
) -> Dict[str, Tuple[float, float]]:
    nodes = [n for n in positions if n in node_communities]
    if len(nodes) < 2:
        return positions

    pos_arr = np.array([positions[n] for n in nodes], dtype=np.float64)
    comms = [node_communities[n] for n in nodes]

    centroids: Dict[int, np.ndarray] = {}
    for comm in set(comms):
        mask = [i for i, c in enumerate(comms) if c == comm]
        centroids[comm] = pos_arr[mask].mean(axis=0)
    global_c = pos_arr.mean(axis=0)

    enhanced: Dict[str, Tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        comm_c = centroids[comms[i]]
        direction = comm_c - global_c
        local_offset = pos_arr[i] - comm_c
        new_pos = global_c + direction * separation_factor + local_offset
        enhanced[node] = (float(new_pos[0]), float(new_pos[1]))

    for n, xy in positions.items():
        if n not in enhanced:
            enhanced[n] = xy
    return enhanced


# ── Normalise to [-1, 1] ──────────────────────────────────────────────────────

def _normalise(positions: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Normalise all positions to approximately [-1, 1] on each axis."""
    if not positions:
        return positions
    xs = [v[0] for v in positions.values()]
    ys = [v[1] for v in positions.values()]
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0
    half_range = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9) / 2.0
    return {
        k: ((v[0] - cx) / half_range, (v[1] - cy) / half_range)
        for k, v in positions.items()
    }


# ── Fallback: kamada-kawai / spring ──────────────────────────────────────────

def _fallback_layout(G: nx.Graph) -> Dict[str, Tuple[float, float]]:
    n = G.number_of_nodes()
    if n == 0:
        return {}
    G_simple = nx.Graph(G.to_undirected() if G.is_directed() else G)
    try:
        if n <= 150:
            pos = nx.kamada_kawai_layout(G_simple, weight="weight")
        else:
            k = 2.5 / math.sqrt(n)
            pos = nx.spring_layout(G_simple, weight="weight", seed=42, k=k, iterations=30)
    except Exception:
        pos = nx.random_layout(G_simple, seed=42)
    return {str(node): (float(x), float(y)) for node, (x, y) in pos.items()}


# ── Fill missing nodes near centroid ─────────────────────────────────────────

def _fill_missing(
    positions: Dict[str, Tuple[float, float]],
    all_node_strs: set,
) -> Dict[str, Tuple[float, float]]:
    missing = all_node_strs - set(positions.keys())
    if not missing:
        return positions
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    cx = float(np.mean(xs)) if xs else 0.0
    cy = float(np.mean(ys)) if ys else 0.0
    spread = max(max(xs) - min(xs), max(ys) - min(ys), 100) * 0.05 if xs else 0.1
    rng = np.random.default_rng(42)
    result = dict(positions)
    for ns in missing:
        result[ns] = (
            float(cx + rng.uniform(-spread, spread)),
            float(cy + rng.uniform(-spread, spread)),
        )
    return result


# ── Public entry point ────────────────────────────────────────────────────────

def compute_vos_layout(
    G: nx.Graph,
    network_type: str,
    papers_df=None,
    embeddings_array: Optional[np.ndarray] = None,
    doc_topics_df=None,
    entities_df=None,
) -> Dict[str, Tuple[float, float]]:
    """
    Two-stage UMAP + ForceAtlas2 VOSviewer-style layout.

    Returns {str(node): (x, y)} normalised to ~[-1, 1].
    Falls back to kamada_kawai/spring when embeddings are absent or on error.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}

    node_strs = {str(nd) for nd in G.nodes()}

    # UMAP needs many data points to produce meaningful positions.
    # For small graphs use the spring/kamada-kawai fallback which keeps
    # all nodes visible and well-distributed.
    if n < 15:
        return _fallback_layout(G)

    # ── Fallback when no embeddings ───────────────────────────────────────────
    if embeddings_array is None or papers_df is None or len(embeddings_array) == 0:
        logger.info("No embeddings; fallback layout for %s", network_type)
        return _fallback_layout(G)

    # ── Stage 1a: per-node embeddings ─────────────────────────────────────────
    logger.info("Computing per-node embeddings for %s (%d nodes)", network_type, n)
    try:
        if network_type == "coauth":
            node_embs = _node_embeddings_coauth(G, papers_df, embeddings_array)
        elif network_type == "keyword":
            node_embs = _node_embeddings_keyword(G, papers_df, embeddings_array)
        elif network_type == "topic":
            node_embs = _node_embeddings_topic(G, papers_df, embeddings_array, doc_topics_df)
        elif network_type == "kg":
            node_embs = _node_embeddings_kg(G, papers_df, embeddings_array, entities_df)
        else:
            node_embs = {}
    except Exception as exc:
        logger.warning("Node embedding computation failed (%s): %s", network_type, exc)
        node_embs = {}

    if len(node_embs) / max(n, 1) < 0.3:
        logger.info("Embedding coverage <30%% for %s; using fallback layout", network_type)
        return _fallback_layout(G)

    # ── Stage 1b: UMAP ────────────────────────────────────────────────────────
    logger.info("Running UMAP for %s (%d embedded nodes)", network_type, len(node_embs))
    umap_pos = _run_umap(node_embs)

    if not umap_pos:
        logger.info("UMAP returned empty; using fallback layout for %s", network_type)
        return _fallback_layout(G)

    # Fill nodes without embeddings near the cluster centroid
    umap_pos = _fill_missing(umap_pos, node_strs)

    # ── Stage 2: ForceAtlas2 ──────────────────────────────────────────────────
    logger.info("Running ForceAtlas2 for %s", network_type)
    fa2_pos = _run_forceatlas2(G, umap_pos)

    # ── Post-process: cluster separation ─────────────────────────────────────
    node_communities = {str(nd): d.get("community", 0) for nd, d in G.nodes(data=True)}
    final_pos = _enhance_cluster_separation(fa2_pos, node_communities, separation_factor=1.4)

    # ── Normalise to [-1, 1] ──────────────────────────────────────────────────
    final_pos = _normalise(final_pos)

    return final_pos
