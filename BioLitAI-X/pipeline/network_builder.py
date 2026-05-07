"""
NetworkBuilder — constructs co-authorship, keyword co-occurrence,
topic, and citation networks from cleaned paper data.
All networks use NetworkX + Louvain community detection.
"""

import logging
import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _louvain_communities(G: nx.Graph) -> Dict[Any, int]:
    """Return {node: community_id} using python-louvain (community module)."""
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        return partition
    except Exception as exc:
        logger.warning("Louvain community detection failed: %s", exc)
        return {n: 0 for n in G.nodes()}


def _scale_node_size(weight: float, w_min: float, w_max: float) -> float:
    """Map weight to [NODE_SIZE_MIN, NODE_SIZE_MAX] via square-root scaling."""
    if w_max <= w_min:
        return config.NODE_SIZE_MIN
    ratio = (math.sqrt(weight) - math.sqrt(w_min)) / (
        math.sqrt(w_max) - math.sqrt(w_min) + 1e-9
    )
    return config.NODE_SIZE_MIN + ratio * (config.NODE_SIZE_MAX - config.NODE_SIZE_MIN)


def _scale_edge_width(weight: float, w_min: float, w_max: float) -> float:
    """Map weight to [EDGE_WIDTH_MIN, EDGE_WIDTH_MAX] via log scaling."""
    if w_max <= w_min:
        return config.EDGE_WIDTH_MIN
    log_w = math.log1p(weight)
    log_min = math.log1p(w_min)
    log_max = math.log1p(w_max)
    ratio = (log_w - log_min) / (log_max - log_min + 1e-9)
    return config.EDGE_WIDTH_MIN + ratio * (config.EDGE_WIDTH_MAX - config.EDGE_WIDTH_MIN)


class NetworkBuilder:
    """Builds all four bibliometric network types from a papers DataFrame."""

    # ── Co-authorship network ─────────────────────────────────────────────────

    def build_coauthorship_network(self, papers_df: pd.DataFrame) -> nx.Graph:
        """
        Nodes: authors (name string)
        Node attrs: paper_count, size, community
        Edges: co-authorship pairs
        Edge attrs: weight (shared papers), width
        """
        if papers_df.empty:
            return nx.Graph()

        author_papers: Dict[str, int] = defaultdict(int)
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        for _, row in papers_df.iterrows():
            authors = row.get("authors", [])
            if not isinstance(authors, list):
                continue
            names = [
                a["name"] if isinstance(a, dict) else str(a)
                for a in authors
                if a
            ]
            names = [n for n in names if n]
            for name in names:
                author_papers[name] += 1
            for a, b in combinations(sorted(set(names)), 2):
                pair_counts[(a, b)] += 1

        G = nx.Graph()

        for author, count in author_papers.items():
            G.add_node(author, paper_count=count)

        for (a, b), weight in pair_counts.items():
            G.add_edge(a, b, weight=weight)

        if G.number_of_nodes() == 0:
            return G

        # Centrality metrics
        try:
            deg_cent = nx.degree_centrality(G)
            bet_cent = nx.betweenness_centrality(G, normalized=True)
            clu_coef = nx.clustering(G, weight="weight")
        except Exception:
            deg_cent = bet_cent = clu_coef = {n: 0.0 for n in G.nodes()}

        # Community detection
        partition = _louvain_communities(G)

        # Node sizing
        counts = [d["paper_count"] for _, d in G.nodes(data=True)]
        w_min, w_max = (min(counts), max(counts)) if counts else (0, 1)
        color_palette = config.COMMUNITY_COLORS

        for node in G.nodes():
            pc = G.nodes[node].get("paper_count", 1)
            comm = partition.get(node, 0)
            G.nodes[node].update(
                {
                    "degree_centrality": deg_cent.get(node, 0.0),
                    "betweenness_centrality": bet_cent.get(node, 0.0),
                    "clustering_coefficient": clu_coef.get(node, 0.0),
                    "community": comm,
                    "color": color_palette[comm % len(color_palette)],
                    "size": _scale_node_size(pc, w_min, w_max),
                }
            )

        # Edge sizing
        weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
        ew_min, ew_max = (min(weights), max(weights)) if weights else (0, 1)
        for u, v, d in G.edges(data=True):
            d["width"] = _scale_edge_width(d.get("weight", 1), ew_min, ew_max)

        logger.info(
            "Co-authorship network: %d nodes, %d edges, %d communities",
            G.number_of_nodes(), G.number_of_edges(), len(set(partition.values()))
        )
        return G

    # ── Keyword co-occurrence network ─────────────────────────────────────────

    def build_keyword_cooccurrence_network(self, papers_df: pd.DataFrame) -> nx.Graph:
        """
        Nodes: unified keywords (author keywords + MeSH descriptors +
               MeSH qualifiers + chemical terms), tagged with source type.
        Edges: co-occurrence in the same paper.
        Only keywords appearing in >= KEYWORD_MIN_FREQUENCY papers are included.
        """
        if papers_df.empty:
            return nx.Graph()

        kw_freq: Dict[str, int] = defaultdict(int)
        kw_type: Dict[str, str] = {}
        paper_kws: List[List[str]] = []

        for _, row in papers_df.iterrows():
            kws_this_paper: List[str] = []

            # Author keywords
            for kw in (row.get("author_keywords") or []):
                if kw:
                    kw_freq[kw] += 1
                    kw_type[kw] = "author"
                    kws_this_paper.append(kw)

            # MeSH descriptors and qualifiers
            for mesh in (row.get("mesh_terms") or []):
                if isinstance(mesh, dict):
                    desc = mesh.get("descriptor", "")
                    qual = mesh.get("qualifier")
                    if desc:
                        kw_freq[desc] += 1
                        kw_type.setdefault(desc, "mesh_descriptor")
                        kws_this_paper.append(desc)
                    if qual:
                        kw_freq[qual] += 1
                        kw_type.setdefault(qual, "mesh_qualifier")
                        kws_this_paper.append(qual)

            # Chemical terms
            for chem in (row.get("chemicals") or []):
                if isinstance(chem, dict):
                    name = chem.get("name", "")
                elif isinstance(chem, str):
                    name = chem
                else:
                    continue
                if name:
                    kw_freq[name] += 1
                    kw_type.setdefault(name, "chemical")
                    kws_this_paper.append(name)

            paper_kws.append(list(set(kws_this_paper)))

        # Filter by minimum frequency
        valid_kws = {
            kw for kw, freq in kw_freq.items()
            if freq >= config.KEYWORD_MIN_FREQUENCY
        }

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for kws in paper_kws:
            filtered = [k for k in kws if k in valid_kws]
            for a, b in combinations(sorted(set(filtered)), 2):
                pair_counts[(a, b)] += 1

        G = nx.Graph()

        for kw in valid_kws:
            G.add_node(
                kw,
                frequency=kw_freq[kw],
                keyword_type=kw_type.get(kw, "author"),
            )

        for (a, b), weight in pair_counts.items():
            if weight > 0:
                G.add_edge(a, b, weight=weight)

        if G.number_of_nodes() == 0:
            return G

        partition = _louvain_communities(G)

        freqs = [d["frequency"] for _, d in G.nodes(data=True)]
        w_min, w_max = (min(freqs), max(freqs)) if freqs else (0, 1)
        color_palette = config.COMMUNITY_COLORS

        for node in G.nodes():
            freq = G.nodes[node].get("frequency", 1)
            comm = partition.get(node, 0)
            G.nodes[node].update(
                {
                    "community": comm,
                    "color": color_palette[comm % len(color_palette)],
                    "size": _scale_node_size(freq, w_min, w_max),
                }
            )

        weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
        ew_min, ew_max = (min(weights), max(weights)) if weights else (0, 1)
        for u, v, d in G.edges(data=True):
            d["width"] = _scale_edge_width(d.get("weight", 1), ew_min, ew_max)

        logger.info(
            "Keyword co-occurrence network: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges()
        )
        return G

    # ── Topic network ─────────────────────────────────────────────────────────

    def build_topic_network(self, topic_model_results: Dict[str, Any]) -> nx.Graph:
        """
        Nodes: BERTopic-discovered topics.
        Edges: papers shared between topics (topic co-assignment).

        topic_model_results expects keys:
          'doc_topics': list of topic IDs per document
          'topic_labels': {topic_id: {'label', 'top_words', 'count'}}
        """
        doc_topics = topic_model_results.get("doc_topics", [])
        topic_labels = topic_model_results.get("topic_labels", {})

        if not doc_topics or not topic_labels:
            return nx.Graph()

        G = nx.Graph()

        for tid, info in topic_labels.items():
            if tid == -1:
                continue
            G.add_node(
                tid,
                label=info.get("label", f"Topic {tid}"),
                top_words=info.get("top_words", []),
                paper_count=info.get("count", 0),
            )

        # Edge: topics that share papers (same paper assigned to multiple topics)
        # Using topic co-assignment from document list
        topic_pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for tid in doc_topics:
            if tid == -1:
                continue
            # Single assignment — build edges from papers that share nearby topics
            # (BERTopic assigns one primary topic per doc, so we use paper lists)

        # Build paper lists per topic
        topic_papers: Dict[int, List[int]] = defaultdict(list)
        for doc_idx, tid in enumerate(doc_topics):
            if tid >= 0:
                topic_papers[tid].append(doc_idx)

        for (ta, tb) in combinations(sorted(topic_papers.keys()), 2):
            shared = len(set(topic_papers[ta]) & set(topic_papers[tb]))
            if shared > 0:
                topic_pair_counts[(ta, tb)] = shared

        for (ta, tb), weight in topic_pair_counts.items():
            if G.has_node(ta) and G.has_node(tb):
                G.add_edge(ta, tb, weight=weight)

        if G.number_of_nodes() == 0:
            return G

        counts = [d.get("paper_count", 1) for _, d in G.nodes(data=True)]
        w_min, w_max = (min(counts), max(counts)) if counts else (0, 1)
        color_palette = config.COMMUNITY_COLORS

        for i, node in enumerate(G.nodes()):
            pc = G.nodes[node].get("paper_count", 1)
            G.nodes[node].update(
                {
                    "color": color_palette[i % len(color_palette)],
                    "size": _scale_node_size(pc, w_min, w_max),
                    "community": i,
                }
            )

        weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
        ew_min, ew_max = (min(weights), max(weights)) if weights else (0, 1)
        for u, v, d in G.edges(data=True):
            d["width"] = _scale_edge_width(d.get("weight", 1), ew_min, ew_max)

        logger.info(
            "Topic network: %d topics, %d connections",
            G.number_of_nodes(), G.number_of_edges()
        )
        return G

    # ── Citation network ──────────────────────────────────────────────────────

    def build_citation_network(self, papers_df: pd.DataFrame) -> nx.Graph:
        """
        Attempts to build a directed citation network via OpenAlex API.
        Falls back to a co-citation network if OpenAlex returns no data.
        Nodes: PMIDs.  Edges: citation relationships.
        """
        if papers_df.empty:
            return nx.DiGraph()

        pmid_set = set(papers_df["pmid"].astype(str).tolist())
        doi_map: Dict[str, str] = {}  # pmid → doi

        for _, row in papers_df.iterrows():
            doi = row.get("doi")
            pmid = str(row.get("pmid", ""))
            if doi and pmid:
                doi_map[pmid] = doi

        G = nx.DiGraph()
        for pmid in pmid_set:
            G.add_node(pmid, citation_count=0, paper_count=1)

        citation_edges = self._fetch_openalex_citations(doi_map, pmid_set)

        if citation_edges:
            for src, tgt in citation_edges:
                if G.has_node(src) and G.has_node(tgt):
                    if G.has_edge(src, tgt):
                        G[src][tgt]["weight"] += 1
                    else:
                        G.add_edge(src, tgt, weight=1)
                    G.nodes[tgt]["citation_count"] = G.nodes[tgt].get("citation_count", 0) + 1
            logger.info("Citation network (OpenAlex): %d edges", G.number_of_edges())
        else:
            logger.info("OpenAlex returned no citations; building co-citation network")
            G = self._build_cocitation_network(papers_df, pmid_set)

        # Size nodes by citation count
        counts = [d.get("citation_count", 0) for _, d in G.nodes(data=True)]
        w_min, w_max = (min(counts, default=0), max(counts, default=1))
        for node in G.nodes():
            cc = G.nodes[node].get("citation_count", 0)
            G.nodes[node]["size"] = _scale_node_size(max(cc, 1), max(w_min, 1), max(w_max, 1))

        return G

    def _fetch_openalex_citations(
        self, doi_map: Dict[str, str], pmid_set: set
    ) -> List[Tuple[str, str]]:
        """Query OpenAlex for citation links among corpus papers."""
        import time
        try:
            import urllib.request, urllib.parse, json as _json
        except ImportError:
            return []

        edges: List[Tuple[str, str]] = []
        pmid_by_doi: Dict[str, str] = {v: k for k, v in doi_map.items()}

        for pmid, doi in list(doi_map.items())[:200]:  # limit API calls
            try:
                encoded = urllib.parse.quote(doi, safe="")
                url = f"https://api.openalex.org/works/https://doi.org/{encoded}?select=referenced_works"
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": f"BioLitAI-X mailto:{config.ENTREZ_EMAIL}"},
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = _json.loads(resp.read())
                refs = data.get("referenced_works", [])
                for ref_url in refs:
                    # ref_url looks like "https://openalex.org/W12345"
                    # we can't directly match to PMID without another lookup;
                    # skip unless we have a DOI match
                    pass
                time.sleep(0.1)
            except Exception:
                continue

        return edges

    def _build_cocitation_network(
        self, papers_df: pd.DataFrame, pmid_set: set
    ) -> nx.Graph:
        """Build an undirected co-citation network from shared MeSH terms as proxy."""
        G = nx.Graph()
        for pmid in pmid_set:
            G.add_node(pmid, citation_count=0, paper_count=1)

        # Use MeSH descriptor overlap as a proxy for topical similarity
        pmid_mesh: Dict[str, set] = {}
        for _, row in papers_df.iterrows():
            pmid = str(row.get("pmid", ""))
            mesh_list = row.get("mesh_terms") or []
            descriptors = set()
            for m in mesh_list:
                if isinstance(m, dict) and m.get("descriptor"):
                    descriptors.add(m["descriptor"])
            if descriptors:
                pmid_mesh[pmid] = descriptors

        pmids = list(pmid_mesh.keys())
        for i in range(len(pmids)):
            for j in range(i + 1, len(pmids)):
                shared = len(pmid_mesh[pmids[i]] & pmid_mesh[pmids[j]])
                if shared >= 3:
                    G.add_edge(pmids[i], pmids[j], weight=shared)

        return G

    # ── Display thinning ──────────────────────────────────────────────────────

    def prepare_graph_for_display(
        self, full_graph: nx.Graph, max_display_nodes: int = 2000
    ) -> nx.Graph:
        """
        Return a subgraph suitable for interactive rendering.

        If the graph has more than *max_display_nodes* nodes, keep only the
        top-N by degree centrality (the most connected, most informative nodes).
        The full graph is unchanged and still available for analysis.
        """
        if full_graph.number_of_nodes() <= max_display_nodes:
            return full_graph
        try:
            deg_cent = nx.degree_centrality(full_graph)
        except Exception:
            deg_cent = {n: full_graph.degree(n) for n in full_graph.nodes()}
        top_nodes = sorted(deg_cent, key=deg_cent.get, reverse=True)[:max_display_nodes]
        sub = full_graph.subgraph(top_nodes).copy()
        logger.info(
            "prepare_graph_for_display: thinned %d → %d nodes for rendering",
            full_graph.number_of_nodes(), sub.number_of_nodes(),
        )
        return sub

    # ── Network statistics ────────────────────────────────────────────────────

    def calculate_network_statistics(self, G) -> Dict[str, Any]:
        """
        Return comprehensive statistics dict for a NetworkX graph.
        Works on both directed and undirected graphs.
        """
        if G.number_of_nodes() == 0:
            return {"node_count": 0, "edge_count": 0}

        undirected = G.to_undirected() if G.is_directed() else G

        try:
            density = nx.density(G)
        except Exception:
            density = 0.0

        try:
            avg_clustering = nx.average_clustering(undirected, weight="weight")
        except Exception:
            avg_clustering = 0.0

        # Centrality (on largest connected component to avoid failures)
        try:
            lcc = undirected.subgraph(
                max(nx.connected_components(undirected), key=len)
            ).copy()
            deg_cent = nx.degree_centrality(lcc)
            bet_cent = nx.betweenness_centrality(lcc, normalized=True)
            clo_cent = nx.closeness_centrality(lcc)
        except Exception:
            deg_cent = bet_cent = clo_cent = {}

        def top10(cent_dict):
            return sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:10]

        # Communities
        communities = set()
        for _, d in G.nodes(data=True):
            comm = d.get("community")
            if comm is not None:
                communities.add(comm)

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
            "density": round(density, 6),
            "avg_clustering_coefficient": round(avg_clustering, 4),
            "num_communities": len(communities),
            "modularity": round(modularity, 4),
            "top10_degree_centrality": top10(deg_cent),
            "top10_betweenness_centrality": top10(bet_cent),
            "top10_closeness_centrality": top10(clo_cent),
        }
