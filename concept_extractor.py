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


# Named entity types to extract as concepts
# Expanded for educational/historical content
VALID_ENTITY_LABELS = {
    "ORG", "PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE",
    "EVENT", "FAC", "GPE", "LOC", "NORP",
    "PERSON",       # historical figures: Gandhi, Newton, Turing
    "DATE",         # historical dates: 1942, 19th century
}

# Words to always exclude — expanded blacklist for academic/educational transcripts
STOPWORDS_EXTRA = {
    # ── Vague filler nouns ──
    "thing", "things", "stuff", "way", "ways", "lot", "lots", "bit",
    "bunch", "couple", "type", "types", "kind", "kinds", "sort", "form",
    "case", "cases", "area", "areas", "piece", "pieces", "aspect",

    # ── Vague references & pronouns ──
    "something", "everything", "nothing", "anything",
    "anyone", "everyone", "someone", "somebody",
    "one", "ones", "other", "others",

    # ── Common verbs that leak through as nouns ──
    "going", "want", "need", "make", "look", "let", "say", "see",
    "use", "get", "take", "come", "know", "think", "try", "start",
    "work", "run", "give", "call", "put", "keep", "set", "turn",
    "show", "talk", "tell", "find", "help", "play", "move",
    "change", "read", "write", "learn", "happen", "feel",

    # ── Adjectives / adverbs that sneak in ──
    "good", "great", "really", "pretty", "actually", "basically",
    "much", "many", "more", "most", "less", "least", "different",
    "same", "whole", "entire", "certain", "important", "interesting",
    "simple", "easy", "hard", "big", "small", "little", "long",
    "high", "low", "real", "right", "wrong", "true", "false",
    "first", "last", "next", "new", "old", "main", "general",
    "specific", "particular", "special", "common", "basic",

    # ── Temporal / ordinal words ──
    "time", "times", "minute", "minutes", "second", "seconds",
    "hour", "hours", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "today", "tomorrow",
    "beginning", "end", "middle", "start", "top", "bottom",

    # ── Structural / meta words (lecture artifacts) ──
    "example", "examples", "question", "questions", "answer", "answers",
    "problem", "problems", "solution", "solutions", "exercise",
    "part", "parts", "course", "courses", "tutorial", "tutorials",
    "chapter", "chapters", "section", "sections", "lesson", "lessons",
    "lecture", "lectures", "class", "classes", "assignment",
    "slide", "slides", "screen", "page", "pages",
    "video", "videos", "clip", "channel",
    "step", "steps", "point", "points", "note", "notes",
    "number", "numbers", "line", "lines", "side", "sides",

    # ── People / audience references ──
    "people", "person", "guys", "guy", "folk", "folks",
    "student", "students", "viewer", "viewers",
    "teacher", "professor", "instructor",
    "friend", "friends", "team", "group",

    # ── Academic filler — the real ghosts ──
    "idea", "ideas", "concept", "concepts", "topic", "topics",
    "definition", "explanation", "description", "introduction",
    "overview", "summary", "review", "conclusion",
    "approach", "method", "technique", "process",
    "result", "results", "output", "input",
    "reason", "purpose", "goal", "objective",
    "fact", "detail", "details", "information",
    "term", "terms", "word", "words", "name", "names",
    "level", "version", "feature", "option", "property",
    "order", "rule", "rules", "pattern", "practice",
    "sense", "difference", "relationship", "connection",
    "reference", "context", "background", "basis",

    # ── Presentation / discourse markers ──
    "hand", "place", "world", "life", "head",
    "eye", "face", "mind", "body", "rest",
    "matter", "issue", "situation", "condition",
}


# Generic everyday nouns that aren't real concepts — the "ghosts" that
# slip past STOPWORDS_EXTRA because they're technically nouns
GENERIC_NOUNS = {
    "year", "month", "week", "day", "hour", "minute", "second",
    "life", "death", "man", "woman", "child", "people", "person",
    "table", "room", "house", "building", "door", "window", "wall",
    "water", "food", "money", "country", "city", "state", "land",
    "road", "street", "car", "book", "paper", "letter", "law",
    "government", "power", "force", "army", "war", "peace",
    "king", "leader", "son", "daughter", "father", "mother",
    "brother", "sister", "family", "home", "school", "plan",
    "act", "bill", "report", "speech", "meeting", "party",
    "member", "officer", "head", "hand", "foot", "eye",
    "face", "voice", "service", "market", "condition",
    "action", "decision", "period", "event", "system",
    "century", "decade", "age", "era", "unit", "rate",
    "growth", "demand", "supply", "cost", "price", "tax",
    "trade", "article", "provision", "right", "claim",
}


def _normalize_concept(text: str) -> str:
    """Normalize a concept string: lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().strip().split())


def _is_valid_concept(concept: str, is_single: bool = False) -> bool:
    """
    Check if a concept passes quality filters.
    
    Single-word concepts face stricter filtering than phrases.
    """
    if len(concept) < 3:
        return False
    if len(concept.split()) > 4:
        return False
    if concept.replace(" ", "").isdigit():
        return False
    
    words = concept.split()
    
    # Reject if the concept itself is a stopword
    if concept in STOPWORDS_EXTRA:
        return False
    
    # Reject phrases where EVERY word is a stopword
    if all(w in STOPWORDS_EXTRA for w in words):
        return False
    
    # Reject phrases where the HEAD (last) noun is a generic/stop word
    # e.g., "large table" → head is "table" → reject
    if len(words) >= 2:
        head = words[-1]
        if head in STOPWORDS_EXTRA or head in GENERIC_NOUNS:
            return False
    
    # Single-word concepts face extra scrutiny
    if is_single or len(words) == 1:
        word = words[0]
        if len(word) < 4:
            return False
        if word in GENERIC_NOUNS:
            return False
    
    return True


def _has_real_noun(tokens) -> bool:
    """Check that a token list contains at least one NOUN or PROPN."""
    return any(t.pos_ in ("NOUN", "PROPN") for t in tokens)


def _absorb_single_words(concept_counter: Counter, phrase_boost: int = 2) -> Counter:
    """
    Post-process concepts: boost multi-word phrases and absorb single-word
    concepts that are substrings of a higher-frequency phrase.
    
    Example: if "binary tree" (freq 20) exists, "binary" (freq 15) and
    "tree" (freq 25) get their counts merged into "binary tree" and are
    removed as standalone concepts.
    """
    # Separate phrases (2+ words) from singles
    phrases = {c: f for c, f in concept_counter.items() if len(c.split()) >= 2}
    singles = {c: f for c, f in concept_counter.items() if len(c.split()) == 1}
    
    merged = Counter()
    absorbed = set()
    
    # Apply phrase boost and absorb singles
    for phrase, freq in phrases.items():
        phrase_words = set(phrase.split())
        boosted_freq = freq * phrase_boost
        
        # Absorb single-word concepts that are part of this phrase
        for single, single_freq in singles.items():
            if single in phrase_words and single not in absorbed:
                boosted_freq += single_freq
                absorbed.add(single)
        
        merged[phrase] = boosted_freq
    
    # Keep singles that weren't absorbed — but only if freq >= 2
    for single, freq in singles.items():
        if single not in absorbed and freq >= 2:
            merged[single] = freq
    
    return merged


def extract_concepts(text: str) -> list[tuple[str, int]]:
    """
    Extract key concepts from a text using spaCy with lemmatization,
    noun-phrase prioritization, and strict filtering.
    
    Strategy:
        1. Noun chunks (phrases) — primary source, e.g. "Quit India Movement"
        2. Named entities — proper nouns, events, orgs, locations
        3. Single nouns — heavily filtered fallback for standalone terms
        4. Absorption — single words merge into their parent phrases
        5. Phrase boost — multi-word concepts get a 2x frequency multiplier
    
    Returns a sorted list of (concept, frequency) tuples.
    """
    doc = nlp(text)
    concept_counter = Counter()
    
    # ── 1. Noun phrases (PRIMARY) ──
    for chunk in doc.noun_chunks:
        # Strip determiners, pronouns, adpositions, conjunctions, punctuation
        tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON", "ADP", "CCONJ", "PUNCT", "SPACE")]
        if not tokens:
            continue
        
        # Must contain at least one real noun (not just adjectives)
        if not _has_real_noun(tokens):
            continue
        
        # Lemmatize each token in the phrase
        concept = _normalize_concept(" ".join(t.lemma_ for t in tokens))
        if _is_valid_concept(concept):
            concept_counter[concept] += 1
    
    # ── 2. Named entities (expanded labels for educational content) ──
    for ent in doc.ents:
        if ent.label_ in VALID_ENTITY_LABELS:
            concept = _normalize_concept(ent.text)
            if _is_valid_concept(concept):
                concept_counter[concept] += 1
    
    # ── 3. Single nouns (strict fallback) ──
    for token in doc:
        if (token.pos_ in ("NOUN", "PROPN") 
            and not token.is_stop 
            and len(token.lemma_) > 4):   # stricter: 5+ chars
            concept = _normalize_concept(token.lemma_)
            if _is_valid_concept(concept, is_single=True):
                concept_counter[concept] += 1
    
    # ── 4. Absorb singles into phrases + apply boost ──
    concept_counter = _absorb_single_words(concept_counter)
    
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


def get_global_concepts(transcripts: list[dict], top_n: int = 50) -> list[tuple[str, int, int]]:
    """
    Aggregate concepts across all videos and return the top N,
    weighted by video spread (how many distinct videos mention each concept).
    
    A concept in 5 videos ranks higher than one in 1 video.
    
    Returns list of (concept, importance_score, video_count) tuples.
    Importance = raw_frequency × video_count (spread multiplier).
    """
    global_counter = Counter()    # concept → total frequency
    video_counter = Counter()     # concept → number of distinct videos
    
    for t in transcripts:
        seen_in_video = set()
        for concept, freq in t.get("concepts", []):
            global_counter[concept] += freq
            if concept not in seen_in_video:
                video_counter[concept] += 1
                seen_in_video.add(concept)
    
    # Compute importance: raw frequency × video spread
    importance = {}
    for concept, freq in global_counter.items():
        spread = video_counter.get(concept, 1)
        importance[concept] = freq * spread
    
    # Sort by importance score, return top N
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return [(concept, score, video_counter.get(concept, 1)) for concept, score in ranked]

