"""
Note Generator Module
Uses Google Gemini API with a chunked summarization pipeline.

Pipeline:
    1. Split transcripts into chunks (1000–1500 words)
    2. Summarize each chunk (small Gemini call)
    3. Merge summaries per video
    4. Generate final notes from merged summaries + concept hierarchy
"""

import os
from dotenv import load_dotenv
from google import genai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Load .env file (contains GEMINI_API_KEY)
load_dotenv()

console = Console()

CHUNK_SIZE = 1200       # target words per chunk
CHUNK_OVERLAP = 100     # overlap words between chunks for context continuity


# ─── Step 0: API Key ────────────────────────────────────────────

def _get_api_key() -> str:
    """Get the Gemini API key from environment variable (.env file)."""
    return os.environ.get("GEMINI_API_KEY", "")


# ─── Step 1: Chunk Transcripts ──────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-based chunks of ~chunk_size words.
    Overlap ensures context isn't lost at chunk boundaries.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # slide back by overlap amount

    return chunks


# ─── Step 2: Summarize Each Chunk ───────────────────────────────

def _summarize_chunk(client, chunk: str, chunk_idx: int, total_chunks: int, video_id: str) -> str:
    """
    Summarize a single transcript chunk using Gemini.
    Returns a condensed summary (~200-300 words).
    """
    prompt = f"""You are summarizing part {chunk_idx + 1}/{total_chunks} of a YouTube video transcript (video: {video_id}).

Extract ONLY the key information:
- Main concepts and definitions explained
- Important relationships between ideas
- Any examples or analogies used
- Technical terms introduced

Keep your summary concise (150-250 words). Focus on substance, skip filler.

Transcript chunk:
{chunk}

Summary:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"[Chunk {chunk_idx + 1} failed: {e}]"


# ─── Step 3: Merge Summaries ───────────────────────────────────

def _merge_summaries(client, summaries: list[str], video_id: str) -> str:
    """
    Merge multiple chunk summaries into a single coherent video summary.
    Only called if there are 3+ chunk summaries (otherwise just join them).
    """
    if len(summaries) <= 2:
        return "\n\n".join(summaries)

    combined = "\n\n---\n\n".join(
        f"**Part {i + 1}:** {s}" for i, s in enumerate(summaries)
    )

    prompt = f"""Below are summaries of different parts of YouTube video {video_id}.
Merge them into ONE coherent summary that:
- Removes redundancy
- Preserves all key concepts and definitions
- Maintains logical flow
- Stays under 500 words

Part summaries:
{combined}

Merged summary:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception:
        # Fallback: just concatenate
        return "\n\n".join(summaries)


# ─── Step 4: Generate Final Notes ──────────────────────────────

def _build_final_prompt(
    level_groups: dict,
    video_summaries: list[dict],
    ordered_concepts: list[tuple],
) -> str:
    """Build the final prompt using condensed summaries instead of raw transcripts."""
    # Build concept hierarchy
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

    # Build video summaries section (much smaller than raw transcripts!)
    summary_sections = []
    for vs in video_summaries:
        summary_sections.append(f"--- Video {vs['video_id']} ---\n{vs['summary']}")

    summaries_text = "\n\n".join(summary_sections)

    prompt = f"""You are an expert educator. I've analyzed {len(video_summaries)} YouTube videos.
Below are CONDENSED SUMMARIES (not raw transcripts) and the extracted concept hierarchy.

## Concept Hierarchy (Prerequisite Order)

{hierarchy_text}

## Video Summaries

{summaries_text}

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


# ─── Public API ────────────────────────────────────────────────

def generate_notes(
    level_groups: dict,
    transcripts: list[dict],
    ordered_concepts: list[tuple],
    api_key: str | None = None,
) -> str:
    """
    Generate structured study notes using a chunked summarization pipeline.

    Pipeline:
        1. Split each transcript into ~1200-word chunks
        2. Summarize each chunk individually (small API calls)
        3. Merge chunk summaries per video
        4. Generate final notes from merged summaries + concept hierarchy

    This uses ~60-80% fewer tokens than sending raw transcripts.
    """
    if not api_key:
        api_key = _get_api_key()

    if not api_key:
        return "❌ No API key provided. Skipping note generation."

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # ── Phase 1: Chunk + Summarize each video ──
    console.print("\n[bold cyan]🤖 Phase 1: Summarizing transcripts in chunks...[/bold cyan]\n")

    video_summaries = []
    total_chunks = 0
    total_words = 0

    for t in transcripts:
        text = t.get("cleaned_text", t.get("full_text", ""))
        video_id = t.get("video_id", "unknown")
        word_count = len(text.split())
        total_words += word_count

        chunks = _chunk_text(text)
        total_chunks += len(chunks)

        console.print(
            f"  [dim]Video {video_id}:[/dim] {word_count:,} words → "
            f"{len(chunks)} chunk{'s' if len(chunks) != 1 else ''}"
        )

        # Summarize each chunk with progress
        chunk_summaries = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold]{task.completed}/{task.total}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"  Summarizing {video_id}...", total=len(chunks))

            for idx, chunk in enumerate(chunks):
                summary = _summarize_chunk(client, chunk, idx, len(chunks), video_id)
                chunk_summaries.append(summary)
                progress.advance(task)

        # Merge chunk summaries into one video summary
        merged = _merge_summaries(client, chunk_summaries, video_id)
        video_summaries.append({"video_id": video_id, "summary": merged})
        console.print(f"  [green]✓ {video_id} summarized[/green]")

    summary_words = sum(len(vs["summary"].split()) for vs in video_summaries)
    console.print(
        f"\n  [dim]📊 {total_words:,} words → {summary_words:,} words "
        f"({100 - (summary_words / max(total_words, 1) * 100):.0f}% reduction)[/dim]"
    )

    # ── Phase 2: Generate final notes from summaries ──
    console.print("\n[bold cyan]🤖 Phase 2: Generating study notes...[/bold cyan]\n")

    prompt = _build_final_prompt(level_groups, video_summaries, ordered_concepts)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
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
