"""
Note Generator Module
Uses Google Gemini API to generate structured study notes from the knowledge map.
"""

import os
from google import genai
from rich.console import Console

console = Console()


def _get_api_key() -> str:
    """
    Get the Gemini API key from environment variable or prompt the user.
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        key = input("\n🔑 Enter your Gemini API key: ").strip()
    return key


def _build_prompt(
    level_groups: dict,
    transcripts: list[dict],
    ordered_concepts: list[tuple],
) -> str:
    """
    Build a detailed prompt for Gemini to generate structured notes.
    """
    # Build concept hierarchy section
    hierarchy_lines = []
    for level in sorted(level_groups.keys()):
        concepts = level_groups[level]
        concept_names = [c[0] for c in concepts]
        if level == 0:
            label = "Foundational"
        elif level == 1:
            label = "Intermediate"
        elif level == 2:
            label = "Advanced"
        else:
            label = f"Level {level}"
        hierarchy_lines.append(f"**{label} (Level {level}):** {', '.join(concept_names)}")
    
    hierarchy_text = "\n".join(hierarchy_lines)
    
    # Build transcript excerpts (first 2000 chars per video to stay within limits)
    excerpts = []
    for i, t in enumerate(transcripts):
        text = t.get("cleaned_text", t.get("full_text", ""))[:2000]
        video_id = t.get("video_id", f"Video {i+1}")
        excerpts.append(f"--- Video {i+1} ({video_id}) ---\n{text}")
    
    excerpts_text = "\n\n".join(excerpts)
    
    prompt = f"""You are an expert educator. I've analyzed transcripts from {len(transcripts)} YouTube videos and extracted key concepts organized by prerequisite level.

## Concept Hierarchy (Prerequisite Order)

{hierarchy_text}

## Transcript Excerpts

{excerpts_text}

## Your Task

Generate **comprehensive, structured study notes** based on the above. Follow these rules:

1. **Organize by prerequisite level** — start with foundational concepts, build up to advanced
2. **For each concept**, provide:
   - A clear, concise **definition** (1-2 sentences)
   - **Why it matters** and how it connects to other concepts
   - A brief **example** if applicable
3. **Show connections** between concepts across levels (e.g., "Understanding X is needed for Y")
4. **Add a summary section** at the end with key takeaways
5. Use **markdown formatting** with headers, bullet points, and bold text
6. Keep the tone educational but approachable

Generate the notes now:"""
    
    return prompt


def generate_notes(
    level_groups: dict,
    transcripts: list[dict],
    ordered_concepts: list[tuple],
    api_key: str | None = None,
) -> str:
    """
    Generate structured study notes using Gemini API.
    
    Returns the generated notes as a markdown string.
    """
    if not api_key:
        api_key = _get_api_key()
    
    if not api_key:
        return "❌ No API key provided. Skipping note generation."
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    # Build the prompt
    prompt = _build_prompt(level_groups, transcripts, ordered_concepts)
    
    console.print("\n[bold cyan]🤖 Generating notes with Gemini...[/bold cyan]\n")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        notes = response.text
        return notes
    except Exception as e:
        return f"❌ Error generating notes: {e}"


def save_notes(notes: str, filename: str = "knowledge_notes.md") -> str:
    """
    Save the generated notes to a markdown file.
    Returns the file path.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# 📚 Knowledge Map — Study Notes\n\n")
        f.write("*Generated from YouTube Transcript Analysis*\n\n")
        f.write("---\n\n")
        f.write(notes)
    
    return os.path.abspath(filename)
