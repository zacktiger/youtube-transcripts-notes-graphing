"""
Transcript Fetcher Module
Fetches YouTube video transcripts using youtube-transcript-api v1.x.
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

console = Console()

# Create a single API instance
_api = YouTubeTranscriptApi()


def extract_video_id(url: str) -> str | None:
    """
    Extract the video ID from various YouTube URL formats.
    
    Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID
        - https://youtube.com/shorts/VIDEO_ID
    """
    patterns = [
        r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_id: str) -> dict | None:
    """
    Fetch the transcript for a single video using youtube-transcript-api v1.x.
    
    Returns a dict with 'video_id', 'segments' (list of timed text), 
    and 'full_text' (concatenated transcript string),
    or None if no transcript is available.
    """
    try:
        # v1.x API: use instance .fetch() method
        transcript = _api.fetch(video_id)
        
        # Convert to list of dicts with text, start, duration
        segments = []
        text_parts = []
        for segment in transcript:
            segments.append({
                "text": segment.text,
                "start": segment.start,
                "duration": segment.duration,
            })
            text_parts.append(segment.text)
        
        full_text = " ".join(text_parts)
        
        return {
            "video_id": video_id,
            "segments": segments,
            "full_text": full_text,
        }
    except Exception as e:
        console.print(f"[red]Error fetching transcript for {video_id}: {e}[/red]")
        return None


def fetch_all_transcripts(urls: list[str]) -> list[dict]:
    """
    Fetch transcripts for multiple YouTube URLs with a progress bar.
    
    Returns a list of transcript dicts (skips failed ones).
    """
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching transcripts...", total=len(urls))
        
        for url in urls:
            video_id = extract_video_id(url.strip())
            if not video_id:
                console.print(f"[yellow]⚠ Could not parse URL: {url}[/yellow]")
                progress.advance(task)
                continue
            
            transcript = fetch_transcript(video_id)
            if transcript:
                transcript["url"] = url.strip()
                results.append(transcript)
                console.print(f"[green]✓ Fetched transcript for {video_id}[/green]")
            else:
                console.print(f"[red]✗ No transcript available for {video_id}[/red]")
            
            progress.advance(task)
    
    return results
