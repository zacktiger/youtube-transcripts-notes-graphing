"""
Prerequisite Order Module
Performs topological sorting on the concept graph to produce prerequisite levels.
"""

import networkx as nx


def _break_cycles(G: nx.DiGraph) -> nx.DiGraph:
    """
    Break cycles in the graph by removing the weakest edge in each cycle.
    Returns a new DAG (directed acyclic graph).
    """
    H = G.copy()
    
    while True:
        try:
            cycle = nx.find_cycle(H)
        except nx.NetworkXNoCycle:
            break
        
        # Find the weakest edge in the cycle and remove it
        weakest_edge = min(cycle, key=lambda e: H.edges[e[0], e[1]].get('weight', 1))
        H.remove_edge(weakest_edge[0], weakest_edge[1])
    
    return H


def compute_levels(G: nx.DiGraph) -> list[tuple[int, str, float]]:
    """
    Assign each concept a prerequisite level using topological sort.
    
    Level 0 = foundational (no prerequisites)
    Level 1 = depends on Level 0 concepts
    Level N = depends on concepts up to Level N-1
    
    Returns a list of (level, concept, pagerank_score) sorted by level then score.
    """
    from concept_graph import get_node_scores
    
    # Break cycles to get a DAG
    dag = _break_cycles(G)
    
    # Get PageRank scores from the original graph
    scores = get_node_scores(G)
    
    # Compute levels using longest path from any root
    levels = {}
    topo_order = list(nx.topological_sort(dag))
    
    for node in topo_order:
        predecessors = list(dag.predecessors(node))
        if not predecessors:
            levels[node] = 0
        else:
            levels[node] = max(levels.get(p, 0) for p in predecessors) + 1
    
    # Build result list
    result = []
    for concept, level in levels.items():
        score = scores.get(concept, 0.0)
        result.append((level, concept, score))
    
    # Sort by level (ascending), then by score (descending) within each level
    result.sort(key=lambda x: (x[0], -x[2]))
    
    return result


def get_level_groups(ordered_concepts: list[tuple[int, str, float]]) -> dict[int, list[tuple[str, float]]]:
    """
    Group ordered concepts by their level.
    
    Returns {level: [(concept, score), ...]}
    """
    groups = {}
    for level, concept, score in ordered_concepts:
        if level not in groups:
            groups[level] = []
        groups[level].append((concept, score))
    
    return groups
