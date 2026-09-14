"""
app.py
Interactive Streamlit Dashboard for ContextFlow Engine.
"""

import streamlit as st
import time
from contextflow import ContextFlow

# Page Configuration
st.set_page_config(
    page_title="ContextFlow Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ ContextFlow Engine Dashboard")
st.caption("Intelligent Caching & Context Optimization Layer for LLMs")

# Sidebar Configuration
st.sidebar.header("Engine Settings")
target_ratio = st.sidebar.slider(
    "Target Compression Ratio (L3)", 
    min_value=0.10, 
    max_value=1.00, 
    value=0.35, 
    step=0.05
)

l2_threshold = st.sidebar.slider(
    "L2 Semantic Threshold", 
    min_value=0.70, 
    max_value=0.98, 
    value=0.85, 
    step=0.01
)

enable_l1 = st.sidebar.checkbox("Enable L1 Exact Cache (SHA-256)", value=True)
enable_l2 = st.sidebar.checkbox("Enable L2 Semantic Cache (Vector)", value=True)
enable_l3 = st.sidebar.checkbox("Enable L3 Compressor", value=True)

# Initialize Engine Instance in Session State so Caches Persist Across Reruns
if "cf_engine" not in st.session_state:
    st.session_state.cf_engine = ContextFlow(
        target_compression_ratio=target_ratio,
        l2_threshold=l2_threshold,
        enable_l1=enable_l1,
        enable_l2=enable_l2,
        enable_l3=enable_l3
    )
else:
    # Update dynamic settings on slider change
    st.session_state.cf_engine.target_ratio = target_ratio
    st.session_state.cf_engine.enable_l1 = enable_l1
    st.session_state.cf_engine.enable_l2 = enable_l2
    st.session_state.cf_engine.enable_l3 = enable_l3

# Main Layout
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("1. Context & Query Input")
    
    default_text = (
        "ContextFlow is an open-source caching framework designed to reduce token usage, "
        "latency, and cost in LLM applications through multi-layer caching and intelligent compression. "
        "Layer 1 utilizes exact SHA-256 hashing for immediate request matching. Layer 2 relies on "
        "semantic similarity matching using SentenceTransformers. Layer 3 employs chunking, relevance "
        "scoring, and budget-aware context compression to trim long documents prior to inference. "
    )
    
    context_str = st.text_area("Paste background document / long context text here:", value=default_text * 8, height=220)
    user_query = st.text_input("Enter User Question:", value="How does Layer 1 caching work in ContextFlow?")
    run_button = st.button("Run ContextFlow Engine", type="primary")

with col_output:
    st.subheader("2. Real-Time Results & Metrics")

    if run_button and context_str and user_query:
        messages = [
            {"role": "system", "content": context_str},
            {"role": "user", "content": user_query}
        ]

        with st.spinner("Processing request through ContextFlow..."):
            res = st.session_state.cf_engine.chat(messages=messages)
            meta = res["contextflow_meta"]

        # 1. Cache & Optimization Badge Display
        st.write("**Engine Decision Layer:**")
        cache_layer = meta["cache_layer"]
        if "L1_EXACT" in cache_layer:
            st.success("🎯 L1 EXACT CACHE HIT (SHA-256 Match)")
        elif "L2_SEMANTIC" in cache_layer:
            st.info(f"🧠 L2 SEMANTIC CACHE HIT (Similarity Score: {meta.get('similarity_score', 'N/A')})")
        else:
            st.warning("⚡ L3 COMPRESSOR ACTIVE (API Inference Run)")

        # 2. Key Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Input Tokens", meta["input_tokens"])
        m2.metric("Output Tokens", meta["output_tokens"])
        m3.metric("Latency", f"{meta['latency_ms']} ms")

        # 3. Output Response
        st.write("**Model Response:**")
        st.info(res["text"])

        # 4. Detailed Compression Stats (If L3 was triggered)
        if meta.get("compression_stats", {}).get("compressed"):
            c_stats = meta["compression_stats"]
            st.divider()
            st.write("**L3 Compression Breakdown:**")
            st.write(f"- **Original Input Tokens:** {c_stats['original_tokens']}")
            st.write(f"- **Compressed Tokens Sent:** {c_stats['compressed_tokens']}")
            st.write(f"- **Token Savings:** {c_stats['tokens_saved']} tokens ({round((1 - c_stats['compression_ratio']) * 100, 1)}%)")