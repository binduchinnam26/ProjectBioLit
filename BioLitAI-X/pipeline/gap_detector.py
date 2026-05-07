"""
GapDetector — identifies structural, temporal, and cross-domain
research gaps in the knowledge graph without any domain-specific
assumptions. All gap detection is driven entirely by graph structure
and publication date data.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

import config

logger = logging.getLogger(__name__)


class GapDetector:
    """
    Detects research gaps from graph structure and temporal data.
    Query-agnostic: works for any biomedical topic.
    """

    def __init__(self, knowledge_graph=None):
        """
        knowledge_graph: a KnowledgeGraph instance (not a nx.Graph directly).
        Can also be set later via set_graph().
        """
        self._kg = knowledge_graph
        self._structural_gaps: List[Dict] = []
        self._temporal_gaps: List[Dict] = []
        self._crossdomain_gaps: List[Dict] = []

    def set_graph(self, knowledge_graph):
        self._kg = knowledge_graph

    def _get_nx_graph(self) -> Optional[nx.Graph]:
        if self._kg is None:
            return None
        if hasattr(self._kg, "graph"):
            return self._kg.graph
        if isinstance(self._kg, nx.Graph):
            return self._kg
        return None

    # ── Structural gap detection ──────────────────────────────────────────────

    def find_structural_gaps(
        self,
        min_shared_neighbors: int = None,
        top_n: int = None,
    ) -> List[Dict]:
        """
        Find node pairs that:
          a. Share >= min_shared_neighbors common neighbors
          b. Have NO direct edge between them
          c. Both have degree above the median degree

        Ranked by: shared_neighbors × avg(degree_a, degree_b).
        Returns top_n gap pairs with supporting evidence.
        """
        if min_shared_neighbors is None:
            min_shared_neighbors = config.GAP_SHARED_NEIGHBORS_MIN
        if top_n is None:
            top_n = config.GAP_TOP_N

        G_directed = self._get_nx_graph()
        if G_directed is None or G_directed.number_of_nodes() == 0:
            logger.warning("find_structural_gaps: no graph available")
            return []

        G = G_directed.to_undirected()

        degrees = dict(G.degree())
        if not degrees:
            return []

        sorted_degrees = sorted(degrees.values())
        median_degree = sorted_degrees[len(sorted_degrees) // 2]

        # Candidate nodes: degree above median
        candidates = [n for n, d in degrees.items() if d > median_degree]
        if len(candidates) < 2:
            candidates = list(G.nodes())

        gaps: List[Dict] = []

        # Pre-compute neighbor sets for candidates
        neighbor_sets: Dict[Any, Set] = {
            n: set(G.neighbors(n)) for n in candidates
        }

        checked = 0
        for i, node_a in enumerate(candidates):
            for node_b in candidates[i + 1:]:
                if G.has_edge(node_a, node_b):
                    continue

                shared = neighbor_sets[node_a] & neighbor_sets[node_b]
                if len(shared) < min_shared_neighbors:
                    continue

                score = len(shared) * (degrees[node_a] + degrees[node_b]) / 2.0
                gaps.append(
                    {
                        "concept_a": node_a,
                        "concept_b": node_b,
                        "shared_neighbors": sorted(shared),
                        "shared_neighbor_count": len(shared),
                        "degree_a": degrees[node_a],
                        "degree_b": degrees[node_b],
                        "gap_score": score,
                        "gap_type": "structural",
                        "entity_type_a": G_directed.nodes[node_a].get("entity_type", "UNKNOWN")
                        if G_directed.has_node(node_a) else "UNKNOWN",
                        "entity_type_b": G_directed.nodes[node_b].get("entity_type", "UNKNOWN")
                        if G_directed.has_node(node_b) else "UNKNOWN",
                    }
                )
                checked += 1

        gaps.sort(key=lambda x: x["gap_score"], reverse=True)
        self._structural_gaps = gaps[:top_n]
        logger.info(
            "Structural gaps: %d candidates checked, %d gaps found (top %d returned)",
            checked, len(gaps), top_n,
        )
        return self._structural_gaps

    # ── Temporal gap detection ────────────────────────────────────────────────

    def find_temporal_gaps(
        self,
        papers_df: pd.DataFrame,
        cutoff_year: int = 2018,
        recent_year: int = 2020,
        min_early_papers: int = 3,
        max_recent_papers: int = 2,
    ) -> List[Dict]:
        """
        Identify entities that were active before *cutoff_year* but have had
        few papers since *recent_year* — signalling a stalled research line.

        All thresholds are parameter-driven, not domain-specific.
        """
        G_directed = self._get_nx_graph()
        if G_directed is None or papers_df.empty:
            return []

        if "pub_year" not in papers_df.columns:
            logger.warning("find_temporal_gaps: pub_year column missing")
            return []

        # Build entity → {year: paper_count} from graph entity names and paper years
        # We approximate: entity appears in paper → use paper's pub_year
        entity_years: Dict[str, List[int]] = defaultdict(list)

        for _, row in papers_df.iterrows():
            year = row.get("pub_year")
            if not year:
                continue
            try:
                year = int(year)
            except (TypeError, ValueError):
                continue

            abstract = str(row.get("abstract", ""))
            for node in G_directed.nodes():
                if node.lower() in abstract.lower():
                    entity_years[node].append(year)

        gaps: List[Dict] = []
        for entity, years in entity_years.items():
            early = [y for y in years if y < cutoff_year]
            recent = [y for y in years if y >= recent_year]
            if len(early) >= min_early_papers and len(recent) <= max_recent_papers:
                gaps.append(
                    {
                        "concept_a": entity,
                        "concept_b": None,
                        "early_paper_count": len(early),
                        "recent_paper_count": len(recent),
                        "last_year_seen": max(years) if years else None,
                        "gap_score": len(early) - len(recent),
                        "gap_type": "temporal",
                        "entity_type_a": G_directed.nodes[entity].get("entity_type", "UNKNOWN")
                        if G_directed.has_node(entity) else "UNKNOWN",
                        "entity_type_b": None,
                    }
                )

        gaps.sort(key=lambda x: x["gap_score"], reverse=True)
        self._temporal_gaps = gaps[:config.GAP_TOP_N]
        logger.info("Temporal gaps found: %d", len(self._temporal_gaps))
        return self._temporal_gaps

    # ── Cross-domain gap detection ────────────────────────────────────────────

    def find_cross_domain_gaps(self) -> List[Dict]:
        """
        Find entity pairs from DIFFERENT biological entity types that are
        indirectly connected (path of length 2) but have no direct edge.

        These represent potential cross-domain hypotheses.
        """
        G_directed = self._get_nx_graph()
        if G_directed is None or G_directed.number_of_nodes() == 0:
            return []

        G = G_directed.to_undirected()
        gaps: List[Dict] = []

        nodes_by_type: Dict[str, List] = defaultdict(list)
        for node, data in G_directed.nodes(data=True):
            etype = data.get("entity_type", "UNKNOWN")
            nodes_by_type[etype].append(node)

        entity_types = list(nodes_by_type.keys())

        for i, type_a in enumerate(entity_types):
            for type_b in entity_types[i + 1:]:
                nodes_a = nodes_by_type[type_a]
                nodes_b = nodes_by_type[type_b]

                for node_a in nodes_a[:50]:  # limit for performance
                    for node_b in nodes_b[:50]:
                        if node_a == node_b:
                            continue
                        if G.has_edge(node_a, node_b):
                            continue
                        # Check for path of length 2 (one shared neighbor)
                        shared = set(G.neighbors(node_a)) & set(G.neighbors(node_b))
                        if shared:
                            score = len(shared) * (G.degree(node_a) + G.degree(node_b)) / 2.0
                            gaps.append(
                                {
                                    "concept_a": node_a,
                                    "concept_b": node_b,
                                    "shared_neighbors": sorted(shared)[:5],
                                    "shared_neighbor_count": len(shared),
                                    "gap_score": score,
                                    "gap_type": "cross_domain",
                                    "entity_type_a": type_a,
                                    "entity_type_b": type_b,
                                }
                            )

        gaps.sort(key=lambda x: x["gap_score"], reverse=True)
        self._crossdomain_gaps = gaps[:config.GAP_TOP_N]
        logger.info("Cross-domain gaps found: %d", len(self._crossdomain_gaps))
        return self._crossdomain_gaps

    # ── Compile gap report ────────────────────────────────────────────────────

    def compile_gap_report(self) -> List[Dict]:
        """
        Combine all gap types into a single ranked list.
        Structural gaps first (most evidence-based), then cross-domain,
        then temporal. Returns structured gap report for hypothesis generation.
        """
        all_gaps: List[Dict] = []

        # Weight structural gaps highest
        for gap in self._structural_gaps:
            gap = dict(gap)
            gap["rank_weight"] = gap["gap_score"] * 3.0
            all_gaps.append(gap)

        for gap in self._crossdomain_gaps:
            gap = dict(gap)
            gap["rank_weight"] = gap["gap_score"] * 2.0
            all_gaps.append(gap)

        for gap in self._temporal_gaps:
            gap = dict(gap)
            gap["rank_weight"] = gap.get("gap_score", 0) * 1.0
            all_gaps.append(gap)

        # Deduplicate by concept pair
        seen: Set[Tuple] = set()
        unique_gaps: List[Dict] = []
        for gap in sorted(all_gaps, key=lambda x: x.get("rank_weight", 0), reverse=True):
            key = tuple(sorted(
                [str(gap.get("concept_a", "")), str(gap.get("concept_b", "") or "")]
            ))
            if key not in seen:
                seen.add(key)
                unique_gaps.append(gap)

        # Attach supporting PMIDs from shared neighbors
        G_directed = self._get_nx_graph()
        if G_directed is not None:
            for gap in unique_gaps:
                evidence_nodes = gap.get("shared_neighbors", [])
                supporting_pmids: List[str] = []
                for nbr in evidence_nodes[:3]:
                    if G_directed.has_node(nbr):
                        # Edges carry evidence_pmids attribute
                        for _, _, d in G_directed.edges(nbr, data=True):
                            supporting_pmids.extend(d.get("evidence_pmids", []))
                gap["supporting_pmids"] = list(set(supporting_pmids))[:10]

        logger.info("Gap report compiled: %d unique gaps", len(unique_gaps))
        return unique_gaps[:config.GAP_TOP_N]
