Here’s a clean, professional but relaxed README you can paste directly into GitHub:

🎓 YouTube Transcript → Knowledge Map

Turn YouTube videos into structured learning paths.

Paste multiple YouTube links.
Extract concepts from transcripts.
Build a prerequisite knowledge graph.
Generate structured AI study notes.

This project transforms a playlist into a map of ideas instead of a list of videos.

🚀 What It Does

Fetches transcripts from multiple YouTube videos

Cleans and processes text using NLP

Extracts meaningful concepts (noun phrases, filtered + lemmatized)

Builds a concept dependency graph

Orders concepts by prerequisite structure

Generates structured study notes using Gemini (optional)

Provides a clean Streamlit web dashboard

🧠 Why This Exists

YouTube is full of knowledge, but it’s linear.
Learning is not.

Instead of watching videos in order and hoping it makes sense, this tool:

Detects core ideas

Infers conceptual relationships

Builds a structured learning path

It becomes a knowledge engine instead of a playlist viewer.

🛠️ Tech Stack

Python

Streamlit

spaCy

NetworkX

YouTube Transcript API

Gemini API (optional, for notes)

📂 Project Structure
project/
│
├── app.py                # Streamlit dashboard
├── transcript.py         # Transcript fetching
├── concepts.py           # NLP + concept extraction
├── concept_graph.py      # Graph building + ordering
├── notes.py              # Gemini-based notes generation
└── main.py               # Core pipeline logic

⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

🔑 Gemini Setup (Optional)

To enable AI-generated notes:

Create a .env file:

GEMINI_API_KEY=your_api_key_here


Make sure .env is added to .gitignore.

If no key is configured, the app will still generate the knowledge map but skip notes generation.

📊 Output

The system produces:

Concept list ranked by importance

Prerequisite levels

Concept dependency tree

Structured AI study notes (optional)

It supports multiple videos and merges knowledge across them.

🔍 Example Use Cases

Structuring lecture playlists

Breaking down long educational videos

Mapping historical or political analysis content

Converting tutorial series into learning roadmaps

🧩 Future Improvements

Interactive graph visualization (clickable nodes)

PDF export for notes

Playlist auto-import

Semantic edge detection using LLM reasoning

Concept clustering across large channels

💡 Design Philosophy

Keep it modular.
Keep it transparent.
Let Python handle structure.
Let AI assist, not replace, reasoning.
