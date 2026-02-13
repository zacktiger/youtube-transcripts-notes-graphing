"""
Concept Graph Module
Builds a directed dependency graph of concepts using networkx.
"""

import networkx as nx
from collections import Counter, defaultdict


def _build_cooccurrence_map(transcripts: list[dict], window_size: int = 5) -> dict:
    """
    Build a co-occurrence map: for each pair of concepts that appear
    in the same video, track how often they co-occur within a sliding
    window of sentences.
    """
    cooccurrence = Counter()
    
    for t in transcripts:
        concepts = t.get("concepts", [])
        concept_names = [c[0] for c in concepts]
        text = t.get("cleaned_text", "")
        
        # Split text into rough sentence chunks
        sentences = [s.strip() for s in text.replace(".", "\n").split("\n") if s.strip()]
        
        # Build a map: concept → list of sentence indices where it appears
        concept_positions = defaultdict(set)
        for idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            for concept in concept_names:
                if concept in sentence_lower:
                    concept_positions[concept].add(idx)
        
        # Count co-occurrences within window
        concept_list = list(concept_positions.keys())
        for i, c1 in enumerate(concept_list):
            for c2 in concept_list[i + 1:]:
                positions_1 = concept_positions[c1]
                positions_2 = concept_positions[c2]
                # Check if they appear within window_size sentences of each other
                for p1 in positions_1:
                    for p2 in positions_2:
                        if abs(p1 - p2) <= window_size:
                            cooccurrence[(c1, c2)] += 1
                            cooccurrence[(c2, c1)] += 1
                            break
                    else:
                        continue
                    break
    
    return cooccurrence


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
    cooccurrence_threshold: int = 1,
) -> nx.DiGraph:
    """
    Build a directed concept dependency graph.
    
    Nodes: top N concepts (by global frequency)
    Edges: A → B means A is a prerequisite for B
    
    Edge inference:
        1. Temporal ordering: if A appears in an earlier video than B
        2. Co-occurrence: A and B frequently appear together
        3. Frequency weighting: more frequent concepts are more foundational
    """
    from concept_extractor import get_global_concepts
    
    # Get top concepts
    global_concepts = get_global_concepts(transcripts, top_n)
    concept_set = {c[0] for c in global_concepts}
    concept_freq = {c[0]: c[1] for c in global_concepts}
    
    # Build helpers
    temporal_order = _get_temporal_order(transcripts)
    cooccurrence = _build_cooccurrence_map(transcripts)
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes with frequency as attribute
    for concept, freq in global_concepts:
        G.add_node(concept, frequency=freq, first_video=temporal_order.get(concept, 0))
    
    # Add edges based on temporal ordering + co-occurrence
    concept_list = list(concept_set)
    for i, c1 in enumerate(concept_list):
        for c2 in concept_list[i + 1:]:
            # Only connect concepts that co-occur
            cooc_score = cooccurrence.get((c1, c2), 0)
            if cooc_score < cooccurrence_threshold:
                continue
            
            # Determine direction: earlier + more frequent → prerequisite
            t1 = temporal_order.get(c1, 999)
            t2 = temporal_order.get(c2, 999)
            f1 = concept_freq.get(c1, 0)
            f2 = concept_freq.get(c2, 0)
            
            # Score: lower temporal order + higher frequency = more foundational
            score1 = f1 - t1 * 10
            score2 = f2 - t2 * 10
            
            if score1 >= score2:
                G.add_edge(c1, c2, weight=cooc_score)
            else:
                G.add_edge(c2, c1, weight=cooc_score)
    
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
