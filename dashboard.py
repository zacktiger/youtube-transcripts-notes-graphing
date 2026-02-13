"""
YouTube Transcript Scraper — Streamlit Dashboard
=================================================
Run:  streamlit run dashboard.py
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import math, os
from dotenv import load_dotenv

# Load .env file (contains GEMINI_API_KEY)
load_dotenv()

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Knowledge Map",
    page_icon="🎓",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }
    .concept-card {
        background: rgba(30, 27, 60, 0.8);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(10px);
    }
    .level-foundation { color: #4ade80; font-weight: 700; }
    .level-core { color: #60a5fa; font-weight: 700; }
    .level-intermediate { color: #fbbf24; font-weight: 700; }
    .level-advanced { color: #f87171; font-weight: 700; }
    div[data-testid="stMetric"] {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 10px;
        padding: 0.8rem;
    }
    h1, h2, h3 {
        background: linear-gradient(90deg, #c084fc, #818cf8, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Graph builder ───────────────────────────────────────────────
def build_graph_figure(G, scores):
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No concepts to display", showarrow=False,
                           font=dict(size=20, color="#8b5cf6"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)")
        return fig

    pos = nx.spring_layout(G, k=2.5 / math.sqrt(len(G.nodes)), iterations=60, seed=42)

    # Edges
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="rgba(139, 92, 246, 0.3)"),
        hoverinfo="none", mode="lines",
    )

    # Nodes
    node_x, node_y, node_text, node_hover, node_size, node_color = [], [], [], [], [], []
    max_score = max(scores.values()) if scores else 1

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        score = scores.get(node, 0)
        freq = G.nodes[node].get("frequency", 0)
        vid_count = G.nodes[node].get("video_count", 1)
        node_hover.append(
            f"<b>{node}</b><br>Score: {score:.4f}<br>"
            f"Frequency: {freq}<br>Videos: {vid_count}"
        )
        node_size.append(14 + (score / max_score) * 35)
        node_color.append(vid_count)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        textfont=dict(size=10, color="#e2e8f0"),
        hovertext=node_hover, hoverinfo="text",
        marker=dict(
            size=node_size, color=node_color,
            colorscale=[[0, "#6366f1"], [0.5, "#8b5cf6"], [1, "#c084fc"]],
            colorbar=dict(title="Videos", thickness=15, len=0.5),
            line=dict(width=1.5, color="rgba(255,255,255,0.3)"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=20, b=0),
        height=550,
    )
    return fig


# ── Session state init ──────────────────────────────────────────
for key in ["transcripts", "level_groups", "ordered_concepts", "G", "scores", "notes", "map_done"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("# 🎓 YouTube Transcript → Knowledge Map")
st.markdown("*Paste YouTube links, extract concepts, build a prerequisite knowledge graph, and generate AI study notes.*")
st.markdown("---")


# ═══════════════════════════════════════════════════════════════
#  INPUT AREA
# ═══════════════════════════════════════════════════════════════
urls_text = st.text_area(
    "📋 **Paste YouTube URLs** (one per line):",
    height=140,
    placeholder="https://www.youtube.com/watch?v=abc123\nhttps://youtu.be/xyz789\nhttps://www.youtube.com/watch?v=...",
    key="url_input",
)

# Settings row
col_s1, col_s2, _ = st.columns([1, 1, 2])
with col_s1:
    top_n = st.slider("Top N concepts", 10, 100, 50, 5)
with col_s2:
    cooc_threshold = st.slider("Co-occurrence threshold", 1, 5, 2)

# ── Buttons ──
btn_col1, btn_col2, _ = st.columns([1, 1, 2])
with btn_col1:
    btn_map = st.button("🚀 Generate Knowledge Map", type="primary", use_container_width=True)
with btn_col2:
    btn_notes = st.button(
        "📝 Generate Notes",
        use_container_width=True,
        disabled=(st.session_state.map_done is None),
    )

st.markdown("---")


# ═══════════════════════════════════════════════════════════════
#  PIPELINE: Generate Knowledge Map
# ═══════════════════════════════════════════════════════════════
if btn_map:
    urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
    if not urls:
        st.error("❌ Please paste at least one YouTube URL above.")
        st.stop()

    # Step 1: Fetch
    with st.status("📡 Fetching transcripts...", expanded=True) as status:
        from transcript_fetcher import fetch_all_transcripts
        transcripts = fetch_all_transcripts(urls)
        if not transcripts:
            st.error("❌ No transcripts could be fetched. Check your URLs.")
            st.stop()
        for t in transcripts:
            st.write(f"✅ `{t.get('video_id', '?')}` — {len(t.get('full_text', '').split()):,} words")
        status.update(label=f"✅ Fetched {len(transcripts)} transcript(s)", state="complete")

    # Step 2: Clean
    with st.status("🧹 Cleaning transcripts...", expanded=False) as status:
        from text_cleaner import clean_all_transcripts
        transcripts = clean_all_transcripts(transcripts)
        status.update(label="✅ Text cleaned", state="complete")

    # Step 3: Extract Concepts
    with st.status("🔍 Extracting concepts...", expanded=True) as status:
        from concept_extractor import extract_concepts_per_video
        transcripts = extract_concepts_per_video(transcripts)
        for t in transcripts:
            top = t.get("top_concepts", [])[:6]
            st.write(f"`{t.get('video_id', '?')}`: {', '.join(c[0] for c in top)}")
        status.update(label="✅ Concepts extracted", state="complete")

    # Step 4: Build Graph
    with st.status("🕸️ Building concept graph...", expanded=False) as status:
        from concept_graph import build_concept_graph, get_node_scores
        G = build_concept_graph(transcripts, top_n=top_n, cooccurrence_threshold=cooc_threshold)
        scores = get_node_scores(G)
        status.update(label=f"✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", state="complete")

    # Step 5: Prerequisite Order
    with st.status("📐 Computing prerequisite order...", expanded=False) as status:
        from prerequisite_order import compute_levels, get_level_groups
        ordered_concepts = compute_levels(G)
        level_groups = get_level_groups(ordered_concepts)
        status.update(label=f"✅ {len(level_groups)} prerequisite levels", state="complete")

    # Save to state
    st.session_state.transcripts = transcripts
    st.session_state.level_groups = level_groups
    st.session_state.ordered_concepts = ordered_concepts
    st.session_state.G = G
    st.session_state.scores = scores
    st.session_state.map_done = True
    st.session_state.notes = None

    st.success("🎉 Knowledge map generated! See results below.")


# ═══════════════════════════════════════════════════════════════
#  PIPELINE: Generate Notes
# ═══════════════════════════════════════════════════════════════
if btn_notes:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in .env file. Add it and restart.")
        st.stop()

    with st.status("🤖 Generating study notes with Gemini...", expanded=True) as status:
        from note_generator import generate_notes
        notes = generate_notes(
            st.session_state.level_groups,
            st.session_state.transcripts,
            st.session_state.ordered_concepts,
            api_key=api_key,
        )
        st.session_state.notes = notes
        status.update(label="✅ Notes generated!", state="complete")


# ═══════════════════════════════════════════════════════════════
#  OUTPUT AREA
# ═══════════════════════════════════════════════════════════════
if st.session_state.map_done:
    transcripts = st.session_state.transcripts
    level_groups = st.session_state.level_groups
    ordered_concepts = st.session_state.ordered_concepts
    G = st.session_state.G
    scores = st.session_state.scores

    # ── Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    total_words = sum(len(t.get("cleaned_text", "").split()) for t in transcripts)
    col1.metric("📹 Videos", len(transcripts))
    col2.metric("💡 Concepts", len(ordered_concepts))
    col3.metric("🔗 Connections", G.number_of_edges())
    col4.metric("📝 Words", f"{total_words:,}")

    st.markdown("---")

    # ──────────────────────────────────────────────────────────
    #  1. CONCEPT ORDER
    # ──────────────────────────────────────────────────────────
    st.markdown("## 🗺️ Concept Order — Prerequisites First")

    level_labels = {0: "🟢 Foundation", 1: "🔵 Core", 2: "🟡 Intermediate"}

    for level in sorted(level_groups.keys()):
        concepts = level_groups[level]
        label = level_labels.get(level, f"🔴 Level {level}")
        level_class = ["level-foundation", "level-core", "level-intermediate", "level-advanced"][min(level, 3)]

        st.markdown(f"#### {label}")

        rows = []
        for concept, score in concepts:
            vid_count = G.nodes[concept].get("video_count", "—") if concept in G.nodes else "—"
            preds = list(G.predecessors(concept)) if concept in G.nodes else []
            succs = list(G.successors(concept)) if concept in G.nodes else []
            rows.append({
                "Concept": concept,
                "Score": f"{score:.4f}",
                "Videos": vid_count,
                "Depends on": ", ".join(preds[:4]) or "—",
                "Leads to": ", ".join(succs[:4]) or "—",
            })

        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ──────────────────────────────────────────────────────────
    #  2. CONCEPT GRAPH
    # ──────────────────────────────────────────────────────────
    st.markdown("## 🕸️ Concept Graph")
    st.markdown("*Node size = importance · Color intensity = video spread · Hover for details*")

    fig = build_graph_figure(G, scores)
    st.plotly_chart(fig, use_container_width=True, key="main_graph")

    # Edge breakdown
    causal = sum(1 for _, _, d in G.edges(data=True) if d.get("source") == "causal")
    cooc = sum(1 for _, _, d in G.edges(data=True) if d.get("source") == "cooccurrence")
    ec1, ec2 = st.columns(2)
    ec1.metric("🔴 Causal Edges", causal)
    ec2.metric("🔵 Co-occurrence Edges", cooc)

    st.markdown("---")

    # ──────────────────────────────────────────────────────────
    #  3. NOTES
    # ──────────────────────────────────────────────────────────
    st.markdown("## 📝 Study Notes")

    if st.session_state.notes:
        st.markdown(st.session_state.notes)
        st.download_button(
            "⬇️ Download Notes (.md)",
            st.session_state.notes,
            file_name="knowledge_notes.md",
            mime="text/markdown",
        )
    else:
        st.info("💡 Click **Generate Notes** above to create AI study notes.")

else:
    # ── Empty state ──
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🎓</div>
        <h2>Paste URLs above and hit Generate Knowledge Map</h2>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem;">
            The pipeline will fetch transcripts, extract concepts, build a dependency graph,
            and organize everything in prerequisite order.
        </p>
        <div style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 2rem; flex-wrap: wrap;">
            <div class="concept-card" style="width: 160px; text-align: center;">
                <div style="font-size: 1.5rem;">📡</div>
                <div style="font-weight: 600;">Fetch</div>
            </div>
            <div style="color: #8b5cf6; font-size: 1.5rem; display: flex; align-items: center;">→</div>
            <div class="concept-card" style="width: 160px; text-align: center;">
                <div style="font-size: 1.5rem;">🔍</div>
                <div style="font-weight: 600;">Extract</div>
            </div>
            <div style="color: #8b5cf6; font-size: 1.5rem; display: flex; align-items: center;">→</div>
            <div class="concept-card" style="width: 160px; text-align: center;">
                <div style="font-size: 1.5rem;">🕸️</div>
                <div style="font-weight: 600;">Connect</div>
            </div>
            <div style="color: #8b5cf6; font-size: 1.5rem; display: flex; align-items: center;">→</div>
            <div class="concept-card" style="width: 160px; text-align: center;">
                <div style="font-size: 1.5rem;">📝</div>
                <div style="font-weight: 600;">Notes</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
