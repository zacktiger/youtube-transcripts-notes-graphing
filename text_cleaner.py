"""
Text Cleaner Module
Cleans raw YouTube transcript text for NLP processing.
"""

import re


# Common filler words and transcript artifacts to remove
FILLER_WORDS = {
    "um", "uh", "erm", "ah", "oh", "like", "you know",
    "sort of", "kind of", "i mean", "basically", "actually",
    "literally", "right", "okay", "ok", "so yeah",
}


def clean_transcript(raw_text: str) -> str:
    """
    Clean raw transcript text:
    - Remove [Music], [Applause], and similar bracket tags
    - Remove filler words
    - Normalize whitespace
    - Fix common punctuation artifacts
    - Remove excessive repetitions
    """
    text = raw_text
    
    # 1. Remove bracket tags like [Music], [Applause], [Laughter]
    text = re.sub(r'\[.*?\]', '', text)
    
    # 2. Remove HTML entities and tags
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Normalize unicode quotes and dashes
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    
    # 4. Remove filler words (case-insensitive, whole word)
    for filler in FILLER_WORDS:
        pattern = r'\b' + re.escape(filler) + r'\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 5. Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # 6. Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # 7. Remove lines that are just timestamps or numbers
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not re.match(r'^[\d:.\-\s]+$', stripped):
            cleaned_lines.append(stripped)
    text = ' '.join(cleaned_lines)
    
    # 8. Final whitespace normalization
    text = text.strip()
    
    return text


def clean_all_transcripts(transcripts: list[dict]) -> list[dict]:
    """
    Clean the full_text field of each transcript dict in-place.
    Also adds a 'cleaned_text' key.
    """
    for t in transcripts:
        t["cleaned_text"] = clean_transcript(t.get("full_text", ""))
    return transcripts
