"""
Concept Extractor Module
Uses spaCy NLP to extract key concepts from cleaned transcript text.
"""

import spacy
from collections import Counter

# Load spaCy model (small English model)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from rich.console import Console
    Console().print(
        "[red]spaCy model not found. Run: python -m spacy download en_core_web_sm[/red]"
    )
    raise


# Concepts we want to keep (nouns, proper nouns, technical terms)
VALID_ENTITY_LABELS = {
    "ORG", "PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE",
    "EVENT", "FAC", "GPE", "LOC", "NORP",
}

# Words to always exclude
STOPWORDS_EXTRA = {
    "thing", "things", "stuff", "way", "lot", "bit",
    "example", "time", "people", "person", "video",
    "guys", "today", "going", "want", "need", "make",
    "look", "let", "say", "see", "use", "get", "take",
    "come", "know", "think", "good", "great", "really",
    "much", "many", "first", "last", "next", "new",
    "something", "everything", "nothing", "anyone",
    "everyone", "someone", "part", "course", "tutorial",
    "chapter", "section", "minute", "second", "hour",
    "day", "year", "number", "step", "point", "kind",
}


def _normalize_concept(text: str) -> str:
    """Normalize a concept string: lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().strip().split())


def _is_valid_concept(concept: str) -> bool:
    """Check if a concept passes quality filters."""
    if len(concept) < 3:
        return False
    if len(concept.split()) > 4:
        return False
    if concept in STOPWORDS_EXTRA:
        return False
    if concept.replace(" ", "").isdigit():
        return False
    return True


def extract_concepts(text: str) -> list[tuple[str, int]]:
    """
    Extract key concepts from a text using spaCy.
    
    Returns a sorted list of (concept, frequency) tuples.
    """
    doc = nlp(text)
    concept_counter = Counter()
    
    # 1. Extract noun chunks (e.g., "machine learning", "data structure")
    for chunk in doc.noun_chunks:
        # Remove determiners and pronouns from the start
        tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON", "ADP")]
        if tokens:
            concept = _normalize_concept(" ".join(t.text for t in tokens))
            if _is_valid_concept(concept):
                concept_counter[concept] += 1
    
    # 2. Extract named entities
    for ent in doc.ents:
        if ent.label_ in VALID_ENTITY_LABELS:
            concept = _normalize_concept(ent.text)
            if _is_valid_concept(concept):
                concept_counter[concept] += 1
    
    # 3. Extract important single nouns (technical terms)
    for token in doc:
        if (token.pos_ in ("NOUN", "PROPN") 
            and not token.is_stop 
            and len(token.text) > 3):
            concept = _normalize_concept(token.text)
            if _is_valid_concept(concept):
                concept_counter[concept] += 1
    
    # Sort by frequency (descending)
    return concept_counter.most_common()


def extract_concepts_per_video(transcripts: list[dict]) -> list[dict]:
    """
    Extract concepts from each transcript.
    
    Adds a 'concepts' key to each transcript dict:
        [(concept_name, frequency), ...]
    
    Also adds 'top_concepts' — the top 30 concepts by frequency.
    """
    for t in transcripts:
        text = t.get("cleaned_text", t.get("full_text", ""))
        all_concepts = extract_concepts(text)
        t["concepts"] = all_concepts
        t["top_concepts"] = all_concepts[:30]
    
    return transcripts


def get_global_concepts(transcripts: list[dict], top_n: int = 50) -> list[tuple[str, int]]:
    """
    Aggregate concepts across all videos and return the top N.
    """
    global_counter = Counter()
    for t in transcripts:
        for concept, freq in t.get("concepts", []):
            global_counter[concept] += freq
    
    return global_counter.most_common(top_n)
