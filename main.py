"""
YouTube Transcript Scraper — Knowledge Map Generator
=====================================================
Paste YouTube URLs → Get structured knowledge in prerequisite order.

Usage:
    python main.py
    python main.py URL1 URL2 URL3 ...
"""

import sys
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


def display_banner():
    """Display the application banner."""
    banner = Text()
    banner.append("🎓 YouTube Transcript Scraper", style="bold magenta")
    banner.append("\n   Knowledge Map Generator", style="dim cyan")
    console.print(Panel(banner, border_style="bright_magenta", padding=(1, 2)))


def get_urls_from_user() -> list[str]:
    """Get YouTube URLs from user input."""
    console.print(
        "\n[bold cyan]📋 Paste YouTube URLs (one per line).[/bold cyan]"
        "\n[dim]Press Enter on an empty line when done.[/dim]\n"
    )
    urls = []
    while True:
        try:
            line = input("  ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        urls.append(line)
    return urls


def display_knowledge_map(level_groups: dict, scores: dict):
    """Display the knowledge map as a rich table."""
    table = Table(
        title="🗺️  Knowledge Map — Prerequisite Order",
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        title_style="bold bright_cyan",
        header_style="bold white on dark_cyan",
        row_styles=["", "dim"],
    )
    table.add_column("Level", justify="center", style="bold yellow", width=8)
    table.add_column("Concepts", style="white", min_width=50)
    table.add_column("Score", justify="right", style="green", width=10)
    
    level_labels = {0: "🟢 Foundation", 1: "🔵 Core", 2: "🟡 Intermediate"}
    
    for level in sorted(level_groups.keys()):
        concepts = level_groups[level]
        label = level_labels.get(level, f"🔴 Level {level}")
        
        for i, (concept, score) in enumerate(concepts):
            lvl_display = label if i == 0 else ""
            table.add_row(lvl_display, concept, f"{score:.4f}")
        
        # Add separator between levels
        if level < max(level_groups.keys()):
            table.add_row("", "─" * 40, "", style="dim")
    
    console.print()
    console.print(table)


def display_dependency_tree(level_groups: dict):
    """Display a tree view of the concept hierarchy."""
    tree = Tree(
        "🌳 [bold bright_magenta]Concept Dependency Tree[/bold bright_magenta]",
        guide_style="bright_cyan",
    )
    
    level_labels = {0: "🟢 Foundation", 1: "🔵 Core", 2: "🟡 Intermediate"}
    
    for level in sorted(level_groups.keys()):
        concepts = level_groups[level]
        label = level_labels.get(level, f"🔴 Level {level}")
        branch = tree.add(f"[bold]{label}[/bold]")
        for concept, score in concepts:
            branch.add(f"[white]{concept}[/white] [dim](score: {score:.4f})[/dim]")
    
    console.print()
    console.print(tree)


def display_stats(transcripts: list[dict], ordered_concepts: list, level_groups: dict):
    """Display summary statistics."""
    stats = Table(box=box.SIMPLE, show_header=False, border_style="dim")
    stats.add_column("Stat", style="bold cyan")
    stats.add_column("Value", style="white")
    stats.add_row("Videos Processed", str(len(transcripts)))
    stats.add_row("Total Concepts", str(len(ordered_concepts)))
    stats.add_row("Prerequisite Levels", str(len(level_groups)))
    
    total_words = sum(len(t.get("cleaned_text", "").split()) for t in transcripts)
    stats.add_row("Total Words Analyzed", f"{total_words:,}")
    
    console.print()
    console.print(Panel(stats, title="📊 Summary", border_style="bright_cyan"))


def main():
    """Main entry point — orchestrates the full pipeline."""
    display_banner()
    
    # ----- Step 1: Get URLs -----
    if len(sys.argv) > 1:
        urls = [u for u in sys.argv[1:] if u.startswith("http")]
    else:
        urls = get_urls_from_user()
    
    if not urls:
        console.print("[red]No URLs provided. Exiting.[/red]")
        return
    
    console.print(f"\n[bold green]✓ {len(urls)} URL(s) received. Starting pipeline...[/bold green]\n")
    
    # ----- Step 2: Fetch Transcripts -----
    from transcript_fetcher import fetch_all_transcripts
    transcripts = fetch_all_transcripts(urls)
    
    if not transcripts:
        console.print("[red]❌ No transcripts could be fetched. Exiting.[/red]")
        return
    
    # ----- Step 3: Clean Text -----
    console.print("\n[cyan]🧹 Cleaning transcripts...[/cyan]")
    from text_cleaner import clean_all_transcripts
    transcripts = clean_all_transcripts(transcripts)
    console.print("[green]✓ Transcripts cleaned.[/green]")
    
    # ----- Step 4: Extract Concepts -----
    console.print("\n[cyan]🔍 Extracting concepts with NLP...[/cyan]")
    from concept_extractor import extract_concepts_per_video
    transcripts = extract_concepts_per_video(transcripts)
    
    # Show per-video concept preview
    for t in transcripts:
        vid = t.get("video_id", "?")
        top = t.get("top_concepts", [])[:8]
        names = ", ".join(c[0] for c in top)
        console.print(f"  [dim]Video {vid}:[/dim] {names}...")
    
    console.print("[green]✓ Concepts extracted.[/green]")
    
    # ----- Step 5: Build Concept Graph -----
    console.print("\n[cyan]🕸️  Building concept graph...[/cyan]")
    from concept_graph import build_concept_graph, get_node_scores
    G = build_concept_graph(transcripts)
    scores = get_node_scores(G)
    console.print(f"[green]✓ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.[/green]")
    
    # ----- Step 6: Prerequisite Ordering -----
    console.print("\n[cyan]📐 Computing prerequisite order...[/cyan]")
    from prerequisite_order import compute_levels, get_level_groups
    ordered_concepts = compute_levels(G)
    level_groups = get_level_groups(ordered_concepts)
    console.print("[green]✓ Concepts ordered by prerequisite level.[/green]")
    
    # ----- Step 7: Display Results -----
    display_knowledge_map(level_groups, scores)
    display_dependency_tree(level_groups)
    display_stats(transcripts, ordered_concepts, level_groups)
    
    # ----- Step 8: Generate Notes (on user request) -----
    console.print(
        "\n[bold bright_magenta]📝 Ready to generate study notes with Gemini AI.[/bold bright_magenta]"
    )
    
    try:
        response = input("\n  Press Enter to generate notes (or 'q' to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        response = "q"
    
    if response.lower() != "q":
        from note_generator import generate_notes, save_notes
        
        notes = generate_notes(level_groups, transcripts, ordered_concepts)
        
        if notes and not notes.startswith("❌"):
            filepath = save_notes(notes)
            console.print(f"\n[bold green]✅ Notes saved to: {filepath}[/bold green]")
            console.print("\n[bold]── Generated Notes Preview ──[/bold]\n")
            # Show first 2000 chars as preview
            preview = notes[:2000]
            if len(notes) > 2000:
                preview += "\n\n[dim]... (see full notes in the saved file)[/dim]"
            console.print(preview)
        else:
            console.print(f"\n[red]{notes}[/red]")
    
    console.print("\n[bold bright_magenta]🎓 Done! Happy learning! 🚀[/bold bright_magenta]\n")


if __name__ == "__main__":
    main()
