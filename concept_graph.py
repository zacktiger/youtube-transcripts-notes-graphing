"""
Concept Graph Module
Builds a directed dependency graph of concepts using networkx.
"""

import re
import networkx as nx
from collections import Counter, defaultdict


# ── Causal / dependency patterns ──
# Each entry: (regex_pattern, direction)
#   direction = "forward"  means: concept BEFORE the pattern → concept AFTER
#   direction = "reverse"  means: concept AFTER the pattern → concept BEFORE
CAUSAL_PATTERNS = [
    # Forward: A leads to B  →  A is prerequisite for B
    (r'\b{c1}\b.*?\b(?:led to|leads to|lead to)\b.*?\b{c2}\b', "forward"),
    (r'\b{c1}\b.*?\b(?:caused|causes|causing)\b.*?\b{c2}\b', "forward"),
    (r'\b{c1}\b.*?\b(?:resulted in|results in|resulting in)\b.*?\b{c2}\b', "forward"),
    (r'\b{c1}\b.*?\b(?:enables?|enabling)\b.*?\b{c2}\b', "forward"),
    (r'\b{c1}\b.*?\b(?:allows?|allowing)\b.*?\b{c2}\b', "forward"),
    (r'\b{c1}\b.*?\b(?:introduces?|introducing)\b.*?\b{c2}\b', "forward"),
    (r'\b{c1}\b.*?\b(?:is (?:the )?basis (?:for|of))\b.*?\b{c2}\b', "forward"),
    (r'\b{c1}\b.*?\b(?:is (?:a )?foundation (?:for|of))\b.*?\b{c2}\b', "forward"),
    (r'\b{c1}\b.*?\b(?:is needed for|is required for)\b.*?\b{c2}\b', "forward"),
    # Reverse: B depends on A  →  A is prerequisite for B
    (r'\b{c2}\b.*?\b(?:depends on|dependent on|depending on)\b.*?\b{c1}\b', "forward"),
    (r'\b{c2}\b.*?\b(?:requires?|requiring)\b.*?\b{c1}\b', "forward"),
    (r'\b{c2}\b.*?\b(?:relies on|relying on)\b.*?\b{c1}\b', "forward"),
    (r'\b{c2}\b.*?\b(?:because of|due to)\b.*?\b{c1}\b', "forward"),
    (r'\b{c2}\b.*?\b(?:built on|builds on|building on)\b.*?\b{c1}\b', "forward"),
    (r'\b{c2}\b.*?\b(?:based on)\b.*?\b{c1}\b', "forward"),
    (r'\b{c2}\b.*?\b(?:extends?|extending)\b.*?\b{c1}\b', "forward"),
    (r'\b{c2}\b.*?\b(?:uses?|using)\b.*?\b{c1}\b', "forward"),
]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    # Split on sentence-ending punctuation, keeping non-empty results
    raw = re.split(r'[.!?]+', text)
    return [s.strip().lower() for s in raw if s.strip()]


def _build_sentence_cooccurrence(transcripts: list[dict], concept_names: list[str]) -> Counter:
    """
    Build a co-occurrence map using STRICT same-sentence matching only.
    Two concepts are connected only if they appear in the exact same sentence.
    """
    cooccurrence = Counter()

    for t in transcripts:
        text = t.get("cleaned_text", "")
        sentences = _split_sentences(text)

        for sentence in sentences:
            # Find which concepts appear in this sentence
            present = [c for c in concept_names if c in sentence]

            # Connect every pair that shares this sentence
            for i, c1 in enumerate(present):
                for c2 in present[i + 1:]:
                    cooccurrence[(c1, c2)] += 1
                    cooccurrence[(c2, c1)] += 1

    return cooccurrence


def _find_causal_edges(transcripts: list[dict], concept_names: list[str]) -> dict:
    """
    Scan transcripts for causal/dependency language patterns between concepts.
    
    Returns {(prerequisite, dependent): match_count}
    These are DIRECTED edges — the pattern tells us which concept comes first.
    """
    causal_edges = Counter()

    for t in transcripts:
        text = t.get("cleaned_text", "").lower()
        sentences = _split_sentences(text)

        for sentence in sentences:
            # Only check concept pairs that are both in this sentence
            present = [c for c in concept_names if c in sentence]
            if len(present) < 2:
                continue

            for i, c1 in enumerate(present):
                for c2 in present[i + 1:]:
                    for pattern_template, direction in CAUSAL_PATTERNS:
                        pattern = pattern_template.format(
                            c1=re.escape(c1), c2=re.escape(c2)
                        )
                        if re.search(pattern, sentence, re.IGNORECASE):
                            if direction == "forward":
                                causal_edges[(c1, c2)] += 1
                            else:
                                causal_edges[(c2, c1)] += 1
                            break  # One pattern match per pair per sentence

    return causal_edges


def _get_temporal_order(transcripts: list[dict]) -> dict:
    """
    Determine the first video index where each concept appears.
    Concepts introduced earlier are more likely prerequisites.
    """
    first_appearance = {}
    for video_idx, t in enumerate(transcripts):
        for concept, _ in t.get("concepts", []):
            if concept not in first_appearance:
                first_appearance[concept] = video_idx
    return first_appearance


def build_concept_graph(
    transcripts: list[dict],
    top_n: int = 50,
    cooccurrence_threshold: int = 2,
) -> nx.DiGraph:
    """
    Build a directed concept dependency graph.
    
    Nodes: top N concepts (by global frequency)
    Edges: A → B means A is a prerequisite for B
    
    Edge sources (strict):
        1. Same-sentence co-occurrence — concepts must share a sentence
        2. Causal patterns — "led to", "requires", "depends on", etc.
        3. Temporal + frequency heuristic — for direction when no causal pattern
    """
    from concept_extractor import get_global_concepts

    # Get top concepts — now returns (concept, importance_score, video_count)
    global_concepts = get_global_concepts(transcripts, top_n)
    concept_set = {c[0] for c in global_concepts}
    concept_freq = {c[0]: c[1] for c in global_concepts}
    concept_video_count = {c[0]: c[2] for c in global_concepts}
    concept_names = list(concept_set)

    # Build helpers
    temporal_order = _get_temporal_order(transcripts)
    cooccurrence = _build_sentence_cooccurrence(transcripts, concept_names)
    causal_edges = _find_causal_edges(transcripts, concept_names)

    # Create graph
    G = nx.DiGraph()

    # Add nodes with frequency + video spread as attributes
    for concept, freq, vid_count in global_concepts:
        G.add_node(
            concept,
            frequency=freq,
            video_count=vid_count,
            first_video=temporal_order.get(concept, 0),
        )

    # ── Add edges from causal patterns (highest confidence) ──
    for (c1, c2), count in causal_edges.items():
        if c1 in concept_set and c2 in concept_set:
            G.add_edge(c1, c2, weight=count * 3, source="causal")

    # ── Add edges from same-sentence co-occurrence ──
    for i, c1 in enumerate(concept_names):
        for c2 in concept_names[i + 1:]:
            # Skip if already connected by a causal edge
            if G.has_edge(c1, c2) or G.has_edge(c2, c1):
                continue

            cooc_score = cooccurrence.get((c1, c2), 0)
            if cooc_score < cooccurrence_threshold:
                continue

            # Determine direction using: video spread + frequency + temporal order
            # More videos + more frequent + earlier = more foundational
            t1 = temporal_order.get(c1, 999)
            t2 = temporal_order.get(c2, 999)
            f1 = concept_freq.get(c1, 0)
            f2 = concept_freq.get(c2, 0)
            v1 = concept_video_count.get(c1, 1)
            v2 = concept_video_count.get(c2, 1)

            # Video spread is the strongest signal for importance
            score1 = (v1 * 50) + f1 - (t1 * 10)
            score2 = (v2 * 50) + f2 - (t2 * 10)

            if score1 >= score2:
                G.add_edge(c1, c2, weight=cooc_score, source="cooccurrence")
            else:
                G.add_edge(c2, c1, weight=cooc_score, source="cooccurrence")

    return G


def get_node_scores(G: nx.DiGraph) -> dict[str, float]:
    """
    Score nodes using PageRank — higher score = more foundational concept.
    """
    if len(G.nodes) == 0:
        return {}
    try:
        scores = nx.pagerank(G, weight='weight')
    except Exception:
        scores = {n: 1.0 / len(G.nodes) for n in G.nodes}
    return scores
