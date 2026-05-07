"""
KnowledgeGraph — builds a biomedical entity-relationship graph from
NLP-extracted entities and relationships. Supports community detection,
semantic edge augmentation, and neighbourhood exploration.
"""

import json
import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _louvain_communities(G: nx.Graph) -> Dict[Any, int]:
    try:
        import community as community_louvain
        return community_louvain.best_partition(G)
    except Exception as exc:
        logger.warning("Louvain failed: %s", exc)
        return {n: 0 for n in G.nodes()}


def _scale_node_size(weight: float, w_min: float, w_max: float) -> float:
    if w_max <= w_min:
        return config.NODE_SIZE_MIN
    ratio = (math.sqrt(max(weight, 0)) - math.sqrt(max(w_min, 0))) / (
        math.sqrt(max(w_max, 0)) - math.sqrt(max(w_min, 0)) + 1e-9
    )
    return config.NODE_SIZE_MIN + ratio * (config.NODE_SIZE_MAX - config.NODE_SIZE_MIN)


class KnowledgeGraph:
    """
    Builds and manages a biomedical knowledge graph from NLP-extracted
    entities and relationships. Works for any biomedical domain.
    """

    def __init__(self):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entity_pmids: Dict[str, Set[str]] = defaultdict(set)

    # ── Build from entities and relationships ─────────────────────────────────

    def build_from_entities(
        self,
        entities_df: pd.DataFrame,
        relationships_df: pd.DataFrame,
    ) -> nx.MultiDiGraph:
        """
        Construct the knowledge graph from SciSpaCy output.

        Node attrs: entity_type, umls_id, paper_count, size, color, community
        Edge attrs: relationship_type, evidence_pmids, confidence_score
        """
        G = nx.MultiDiGraph()
        self.graph = G
        self._entity_pmids = defaultdict(set)

        if entities_df.empty:
            logger.warning("build_from_entities: empty entities DataFrame")
            return G

        # ── Add nodes ─────────────────────────────────────────────────────────
        entity_paper_count: Dict[str, int] = defaultdict(int)
        entity_info: Dict[str, Dict] = {}

        for _, row in entities_df.iterrows():
            name = row.get("entity_text", "")
            if not name:
                continue
            etype = row.get("entity_type", "UNKNOWN")
            umls_id = row.get("umls_id")
            pmid = str(row.get("pmid", ""))

            entity_paper_count[name] += 1
            self._entity_pmids[name].add(pmid)

            if name not in entity_info:
                entity_info[name] = {"entity_type": etype, "umls_id": umls_id}

        counts = list(entity_paper_count.values())
        w_min = min(counts) if counts else 0
        w_max = max(counts) if counts else 1

        for name, info in entity_info.items():
            etype = info["entity_type"]
            color = config.ENTITY_TYPE_COLORS.get(etype, "#9CA3AF")
            G.add_node(
                name,
                entity_type=etype,
                umls_id=info.get("umls_id"),
                paper_count=entity_paper_count[name],
                color=color,
                size=_scale_node_size(entity_paper_count[name], w_min, w_max),
                is_gap_node=False,
            )

        # ── Community detection on undirected projection ───────────────────────
        undirected = G.to_undirected()
        partition = _louvain_communities(undirected)
        for node, comm in partition.items():
            if G.has_node(node):
                G.nodes[node]["community"] = comm

        # ── Add edges ─────────────────────────────────────────────────────────
        if not relationships_df.empty:
            rel_groups: Dict[tuple, List] = defaultdict(list)
            for _, row in relationships_df.iterrows():
                src = row.get("subject_entity", "")
                tgt = row.get("object_entity", "")
                verb = row.get("relationship_verb", "")
                pmid = str(row.get("pmid", ""))
                if src and tgt and verb and G.has_node(src) and G.has_node(tgt):
                    rel_groups[(src, tgt, verb)].append(pmid)

            for (src, tgt, verb), pmids in rel_groups.items():
                confidence = min(1.0, len(set(pmids)) / 5.0)
                G.add_edge(
                    src, tgt,
                    relationship_type=verb,
                    evidence_pmids=list(set(pmids)),
                    confidence_score=confidence,
                )

        # Re-run community detection now that edges are present
        undirected2 = G.to_undirected()
        partition2 = _louvain_communities(undirected2)
        for node, comm in partition2.items():
            if G.has_node(node):
                G.nodes[node]["community"] = comm

        logger.info(
            "KnowledgeGraph built: %d entities, %d relationships",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    # ── Semantic edge augmentation ────────────────────────────────────────────

    def add_semantic_edges(
        self,
        embeddings: np.ndarray,
        pmid_list: List[str],
        papers_df: pd.DataFrame,
        similarity_threshold: float = None,
    ):
        """
        Add 'semantic_similarity' edges between entities whose source paper
        embeddings are highly similar (above threshold).

        embeddings: array of shape [n_papers, dim] aligned with pmid_list.
        """
        if similarity_threshold is None:
            similarity_threshold = config.SEMANTIC_SIMILARITY_THRESHOLD

        if embeddings is None or len(embeddings) == 0:
            return

        if self.graph.number_of_nodes() == 0:
            return

        # Build pmid → paper entities map
        pmid_idx = {pmid: i for i, pmid in enumerate(pmid_list)}

        # Build entity → pmids map from graph node attributes
        entity_pmids = {
            node: list(self._entity_pmids.get(node, set()))
            for node in self.graph.nodes()
        }

        # Compute mean embedding per entity (average over its source papers)
        entity_vecs: Dict[str, np.ndarray] = {}
        for entity, pmids in entity_pmids.items():
            idxs = [pmid_idx[p] for p in pmids if p in pmid_idx]
            if idxs:
                entity_vecs[entity] = embeddings[idxs].mean(axis=0)

        entities = list(entity_vecs.keys())
        if len(entities) < 2:
            return

        added = 0
        from itertools import combinations
        for e1, e2 in combinations(entities, 2):
            v1, v2 = entity_vecs[e1], entity_vecs[e2]
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm < 1e-9:
                continue
            sim = float(np.dot(v1, v2) / norm)
            if sim >= similarity_threshold:
                self.graph.add_edge(
                    e1, e2,
                    relationship_type="semantic_similarity",
                    evidence_pmids=[],
                    confidence_score=sim,
                )
                added += 1

        logger.info("Added %d semantic similarity edges (threshold=%.2f)", added, similarity_threshold)

    # ── Neighbourhood exploration ─────────────────────────────────────────────

    def get_entity_neighborhood(
        self, entity_name: str, depth: int = 2
    ) -> nx.MultiDiGraph:
        """
        Return a subgraph containing *entity_name* and all nodes within
        *depth* hops (on the undirected projection).
        """
        if not self.graph.has_node(entity_name):
            logger.warning("Entity '%s' not found in graph", entity_name)
            return nx.MultiDiGraph()

        undirected = self.graph.to_undirected(as_view=True)
        reachable: Set[str] = {entity_name}
        frontier = {entity_name}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(undirected.neighbors(node))
            frontier = next_frontier - reachable
            reachable.update(frontier)

        return self.graph.subgraph(reachable).copy()

    # ── Graph statistics ──────────────────────────────────────────────────────

    def get_graph_statistics(self) -> Dict[str, Any]:
        G = self.graph
        if G.number_of_nodes() == 0:
            return {"node_count": 0, "edge_count": 0}

        # Node counts by type
        type_counts: Dict[str, int] = defaultdict(int)
        for _, d in G.nodes(data=True):
            type_counts[d.get("entity_type", "UNKNOWN")] += 1

        # Edge counts by type
        edge_type_counts: Dict[str, int] = defaultdict(int)
        for _, _, d in G.edges(data=True):
            edge_type_counts[d.get("relationship_type", "unknown")] += 1

        # Most connected nodes (by degree in undirected projection)
        undirected = G.to_undirected()
        degree_seq = sorted(undirected.degree(), key=lambda x: x[1], reverse=True)
        most_connected = degree_seq[:10]

        # Isolated nodes
        isolated = [n for n in undirected.nodes() if undirected.degree(n) == 0]

        # Modularity
        modularity = 0.0
        try:
            import community as community_louvain
            partition = {n: G.nodes[n].get("community", 0) for n in G.nodes()}
            modularity = community_louvain.modularity(partition, undirected)
        except Exception:
            pass

        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "node_count_by_type": dict(type_counts),
            "edge_count_by_type": dict(edge_type_counts),
            "most_connected_nodes": most_connected,
            "isolated_node_count": len(isolated),
            "modularity": round(modularity, 4),
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def export_to_json(self) -> str:
        """Export the full graph as a JSON-serialisable string."""
        G = self.graph
        data = {
            "nodes": [],
            "edges": [],
        }

        for node, attrs in G.nodes(data=True):
            node_data = {"id": node}
            for k, v in attrs.items():
                if isinstance(v, set):
                    v = list(v)
                node_data[k] = v
            data["nodes"].append(node_data)

        for src, tgt, attrs in G.edges(data=True):
            edge_data = {"source": src, "target": tgt}
            for k, v in attrs.items():
                if isinstance(v, set):
                    v = list(v)
                edge_data[k] = v
            data["edges"].append(edge_data)

        return json.dumps(data, ensure_ascii=False, indent=2)

    # ── Gap node highlighting ─────────────────────────────────────────────────

    def mark_gap_nodes(self, gap_pairs: List[tuple]):
        """Mark entity nodes that appear in gap pairs with is_gap_node=True."""
        for pair in gap_pairs:
            a, b = pair[0], pair[1]
            for node in (a, b):
                if self.graph.has_node(node):
                    self.graph.nodes[node]["is_gap_node"] = True
