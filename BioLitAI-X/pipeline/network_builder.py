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

        for row in papers_df.to_dict("records"):
            authors = row.get("authors", [])
            if not isinstance(authors, list):
                continue
            names = [
                a["name"] if isinstance(a, dict) else str(a)
                for a in authors
                if a
            ]
            names = [n for n in names if n]
            # Cap authors per paper to avoid O(n²) explosion on papers with 50+ authors
            names = names[:20]
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

        # Degree centrality only — betweenness is O(n²k) and takes 2-5 min on
        # large co-authorship graphs; it is not shown in any display element.
        try:
            deg_cent = nx.degree_centrality(G)
        except Exception:
            deg_cent = {n: 0.0 for n in G.nodes()}

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

        for row in papers_df.to_dict("records"):
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
        Nodes: one node per unique keyword extracted from BERTopic topics.
        Edges: two keywords are connected when they co-appear in the same topic.

        topic_model_results expects keys:
          'doc_topics'   : list of topic IDs per document (required)
          'topic_labels' : {topic_id: {'label', 'top_words', 'count', 'avg_year'}}
          'doc_probs'    : probability matrix (unused here, kept for API compat)
          'embeddings'   : (unused here, kept for API compat)
          'papers_df'    : DataFrame with pmid, pub_year columns (optional)
        """
        doc_topics   = topic_model_results.get("doc_topics", [])
        topic_labels = topic_model_results.get("topic_labels", {})

        if not topic_labels:
            return nx.Graph()

        N_WORDS_PER_TOPIC = 10

        # Count papers per topic from doc_topics list
        topic_paper_count: Dict[int, int] = defaultdict(int)
        for tid in doc_topics:
            if tid >= 0:
                topic_paper_count[tid] += 1

        # Accumulate per-word data across topics
        word_total_weight: Dict[str, float] = defaultdict(float)
        word_topic_ids:    Dict[str, List[int]] = defaultdict(list)
        word_topic_names:  Dict[str, List[str]] = defaultdict(list)

        # topic_id → ordered word list (top N)
        topic_word_lists: Dict[int, List[str]] = {}

        for tid, info in topic_labels.items():
            if tid == -1:
                continue
            words = (info.get("top_words") or info.get("words") or [])[:N_WORDS_PER_TOPIC]
            if not words:
                continue
            pc         = topic_paper_count.get(tid, info.get("count", 1)) or 1
            topic_name = info.get("label", f"Topic {tid}")
            topic_word_lists[tid] = words

            for rank, word in enumerate(words):
                # Higher-ranked words (lower rank index) get more weight
                rank_weight = pc / (rank + 1)
                word_total_weight[word] += rank_weight
                word_topic_ids[word].append(tid)
                word_topic_names[word].append(topic_name)

        G = nx.Graph()

        for word, total_w in word_total_weight.items():
            G.add_node(
                word,
                label=word,
                paper_count=total_w,
                topic_ids=word_topic_ids[word],
                topic_names=word_topic_names[word],
                n_topics=len(word_topic_ids[word]),
            )

        if G.number_of_nodes() < 2:
            return G

        # ── Edges: keywords co-appearing in the same topic ────────────────────
        edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)

        for tid, words in topic_word_lists.items():
            pc = topic_paper_count.get(tid, topic_labels[tid].get("count", 1)) or 1
            n  = len(words)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    key = (min(words[i], words[j]), max(words[i], words[j]))
                    edge_weights[key] += pc  # accumulate paper count across shared topics

        for (w1, w2), weight in edge_weights.items():
            if G.has_node(w1) and G.has_node(w2):
                G.add_edge(w1, w2, weight=weight, relationship_type="co_topic")

        # ── Community detection and node sizing ───────────────────────────────
        partition = _louvain_communities(G)
        weights   = [G.nodes[n].get("paper_count", 1) for n in G.nodes()]
        w_min, w_max = (min(weights), max(weights)) if weights else (1, 1)

        for node in G.nodes():
            pw    = G.nodes[node].get("paper_count", 1)
            comm  = partition.get(node, 0)
            ratio = (pw - w_min) / (w_max - w_min + 1e-9)
            G.nodes[node].update({
                "color":     config.COMMUNITY_COLORS[comm % len(config.COMMUNITY_COLORS)],
                "size":      10.0 + ratio * 50.0,
                "community": comm,
            })

        all_w = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
        ew_min = min(all_w) if all_w else 0.0
        ew_max = max(all_w) if all_w else 1.0
        for u, v, d in G.edges(data=True):
            d["width"] = _scale_edge_width(d.get("weight", 1), ew_min, ew_max)

        logger.info(
            "Topic keyword network: %d keyword nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
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

        # Cap to 500 papers for the O(n²) loop — beyond that it becomes too slow
        pmids = list(pmid_mesh.keys())[:500]
        for i in range(len(pmids)):
            for j in range(i + 1, len(pmids)):
                shared = len(pmid_mesh[pmids[i]] & pmid_mesh[pmids[j]])
                if shared >= 3:
                    G.add_edge(pmids[i], pmids[j], weight=shared)

        return G

    # ── Display thinning ──────────────────────────────────────────────────────

    def prepare_graph_for_display(
        self, full_graph: nx.Graph, max_display_nodes: int = None
    ) -> nx.Graph:
        """
        Return a subgraph suitable for interactive browser rendering.

        Keeps only the top-N most-connected nodes. Default cap comes from
        config.GRAPH_MAX_DISPLAY_NODES (500) which is safe for browser canvas.
        The full graph is unchanged and still available for analysis.
        """
        if max_display_nodes is None:
            max_display_nodes = config.GRAPH_MAX_DISPLAY_NODES

        if full_graph.number_of_nodes() <= max_display_nodes:
            return full_graph
        # Use raw degree (fast, O(n)) instead of degree_centrality (same ranking for subgraph)
        top_nodes = sorted(full_graph.nodes(), key=lambda n: full_graph.degree(n), reverse=True)[:max_display_nodes]
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
            # For large graphs, sample 500 nodes to estimate clustering quickly
            n = undirected.number_of_nodes()
            if n > 500:
                import random
                sample = random.sample(list(undirected.nodes()), 500)
                avg_clustering = nx.average_clustering(
                    undirected, nodes=sample, weight="weight"
                )
            else:
                avg_clustering = nx.average_clustering(undirected, weight="weight")
        except Exception:
            avg_clustering = 0.0

        # Degree centrality on largest connected component only.
        # Betweenness (O(n²k)) and closeness (O(n²)) are skipped — they take
        # 2-10 min on graphs with thousands of authors and are not displayed.
        try:
            lcc = undirected.subgraph(
                max(nx.connected_components(undirected), key=len)
            ).copy()
            deg_cent = nx.degree_centrality(lcc)
        except Exception:
            deg_cent = {}
        bet_cent: dict = {}
        clo_cent: dict = {}

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
