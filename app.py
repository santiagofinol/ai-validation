import streamlit as st
import json
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

st.set_page_config(page_title="AI Reliability Lab", layout="wide")


@st.cache_data
def load_data(results_path="validation_results.json", analysis_path="validation_analysis.json"):
    try:
        with open(results_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Results file not found: `{results_path}`. Run `test_engine.py` first.")
        return None, None
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in `{results_path}`: {e}")
        return None, None

    try:
        with open(analysis_path) as f:
            analysis = json.load(f)
    except FileNotFoundError:
        st.error(f"Analysis file not found: `{analysis_path}`. Run `test_engine.py` first.")
        return data, None
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in `{analysis_path}`: {e}")
        return data, None

    return data, analysis


uploaded_results = st.sidebar.file_uploader("Upload validation_results.json", type="json")
uploaded_analysis = st.sidebar.file_uploader("Upload validation_analysis.json", type="json")

if uploaded_results and uploaded_analysis:
    data = json.load(uploaded_results)
    analysis = json.load(uploaded_analysis)
else:
    data, analysis = load_data()

if data is None:
    st.stop()

outputs = [d["output"] for d in data]
n = len(data)

st.title("AI Output Validation Framework")
st.markdown("*Systems Engineering Consistency Tester — v3.0*")
st.markdown("---")

with st.expander("📖 Methodology — How This Works"):
    st.markdown("""
**TF-IDF (Term Frequency–Inverse Document Frequency)**

Before comparing two texts, each one is converted to a vector of numbers — one number per unique word. Raw word counts would make common words like "the" and "is" dominate. TF-IDF fixes this: it down-weights words that appear in every document and up-weights words that are distinctive (like "overcharge" or "IEC"). The result is a vector that actually represents what the document *is about*.

**Cosine Similarity**

Once we have TF-IDF vectors, we measure the angle between them rather than the distance. Why? Because a longer response covering the same topics should score the same as a shorter one — and cosine similarity is invariant to vector length. Two documents with identical vocabulary and proportions get a score of 1.0; unrelated documents approach 0.0.

**Composite Score Breakdown**

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Text similarity | 40% | Vocabulary consistency across runs |
| Key-term consistency | 25% | Whether critical concepts appear in every run |
| Numeric agreement | 20% | Whether specific values (temperatures, voltages) agree |
| Golden answer match | 15% | Similarity to human-written reference |

**Why 75% threshold?**  
It's an empirically tuned heuristic on this dataset — not an industry standard. Would need calibration against human-labelled examples to be defensible in production.
""")

# =============================================================================
# RELIABILITY GAUGE
# =============================================================================
if analysis:
    sd = analysis.get("score_data", {})
    score = sd.get("total_score")

    if score is None:
        st.warning("Analysis JSON missing 'score_data.total_score'.")
    else:
        if score >= 80:
            color, status, emoji = "#28a745", "HIGH RELIABILITY", "✅"
        elif score >= 60:
            color, status, emoji = "#ffc107", "MODERATE RELIABILITY", "⚠️"
        else:
            color, status, emoji = "#dc3545", "LOW RELIABILITY", "❌"

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align:center;padding:20px;border-radius:15px;
                        background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);">
                <h2 style="color:{color};margin-bottom:5px;">{emoji} {score}%</h2>
                <p style="color:{color};font-size:1.2em;font-weight:bold;">{status}</p>
                <p style="color:#888;font-size:0.9em;">Combined Reliability Score</p>
            </div>
            """, unsafe_allow_html=True)

        explanation = sd.get("score_explanation")
        if explanation:
            st.info(explanation)

# =============================================================================
# METRICS ROW
# =============================================================================
if analysis:
    sd = analysis.get("score_data", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Text Similarity", f"{sd.get('text_similarity', 'N/A')}%")
    c2.metric("Numeric Agreement", f"{sd.get('numeric_agreement', 'N/A')}%")
    c3.metric("Key-Term Consistency", f"{sd.get('key_term_consistency', 'N/A')}%")
    rel = sd.get("relevance_ratio")
    golden = sd.get("golden_answer_similarity")
    if rel is not None:
        c4.metric("Output Relevance", f"{rel}%")
    elif golden is not None:
        c4.metric("Golden Answer Match", f"{golden}%")
    else:
        c4.metric("Total Iterations", n)

st.markdown("---")

# =============================================================================
# HALLUCINATION DETECTION
# =============================================================================
st.subheader("🔍 Hallucination Detection")

if analysis:
    unverified = analysis.get("unverified_standards", [])
    if unverified:
        st.error(f"⚠️ {len(unverified)} unverified standard(s) detected (possible hallucinations)")
        std_data = []
        for i, stds in enumerate(analysis.get("standards_per_run", []), 1):
            row = {"Run": f"Run {i}"}
            for std in unverified:
                row[std] = "✅ Found" if std.upper() in [s.upper() for s in stds] else "❌ Missing"
            std_data.append(row)
        st.dataframe(pd.DataFrame(std_data), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No unverified standards detected across all runs.")

    std_presence = analysis.get("standard_presence", {})
    if std_presence:
        st.markdown("**Verified Standards Detected:**")
        cols = st.columns(min(4, len(std_presence)))
        for idx, (std, count) in enumerate(sorted(std_presence.items(), key=lambda x: -x[1])):
            with cols[idx % 4]:
                pct = count / n * 100
                st.markdown(f"`{std}`<br><small>{count}/{n} runs ({pct:.0f}%)</small>",
                            unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# SEMANTIC CONTRADICTION DETECTION
# =============================================================================
st.subheader("⚡ Semantic Contradiction Detection")

if analysis:
    contradictions = analysis.get("score_data", {}).get("contradictions", [])
    if contradictions:
        st.warning(f"{len(contradictions)} potential contradiction(s) detected between runs.")
        c_rows = [
            {
                "Run A": f"Run {c['run_a'] + 1}",
                "Run B": f"Run {c['run_b'] + 1}",
                "Keyword in A": c["keyword"],
                "Conflicts with (in B)": c["antonym"],
                "Severity": c["severity"],
            }
            for c in contradictions
        ]
        st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
        st.caption(
            "v1 detection uses a keyword-antonym map. "
            "False positives are possible; always verify manually."
        )
    else:
        st.success("✅ No keyword-level contradictions detected.")

st.markdown("---")

# =============================================================================
# REQUIREMENT TRACEABILITY
# =============================================================================
st.subheader("📋 Requirement Traceability Matrix")

if analysis:
    sd = analysis.get("score_data", {})
    shall_counts = sd.get("shall_counts", [])
    total_shalls = sum(shall_counts)

    if total_shalls == 0:
        st.warning("⚠️ No 'shall' statements found. Engineering requirements should use normative language.")

    shall_data = [
        {
            "Run": f"Run {i}",
            "Shall Count": len(shalls),
            "Latency (s)": data[i - 1].get("latency_sec", "N/A"),
            "Status": "✅" if data[i - 1].get("status", "ok") == "ok" else "❌",
        }
        for i, shalls in enumerate(analysis.get("shalls_per_run", []), 1)
    ]
    st.dataframe(pd.DataFrame(shall_data), use_container_width=True, hide_index=True)

    for i, shalls in enumerate(analysis.get("shalls_per_run", []), 1):
        if shalls:
            with st.expander(f"Run {i} — {len(shalls)} shall statement(s)"):
                for s in shalls:
                    st.markdown(f"- {s}")

st.markdown("---")

# =============================================================================
# KEY-TERM HEATMAP
# =============================================================================
st.subheader("🎯 Key-Term Consistency Heatmap")

if analysis:
    term_df_data = []
    for i, terms in enumerate(analysis.get("terms_per_run", []), 1):
        row = {"Run": f"Run {i}"}
        row.update(terms)
        term_df_data.append(row)

    if term_df_data:
        term_df = pd.DataFrame(term_df_data)
        bool_cols = [c for c in term_df.columns if c != "Run"]

        def color_bool(val):
            return "background-color: #28a745" if val else "background-color: #dc3545"

        st.dataframe(
            term_df.style.map(color_bool, subset=bool_cols),
            use_container_width=True,
            hide_index=True,
        )

        sd = analysis.get("score_data", {})
        term_summary = sd.get("term_breakdown", {})
        if term_summary:
            st.markdown("**Term Presence Frequency:**")
            chart_df = pd.DataFrame({
                "Term": list(term_summary.keys()),
                "Presence (%)": list(term_summary.values()),
            })
            st.bar_chart(chart_df.set_index("Term"))

st.markdown("---")

# =============================================================================
# NUMERIC CONSISTENCY + SI-NORMALIZED TABLE
# =============================================================================
st.subheader("🔢 Numeric Consistency")

if analysis:
    sd = analysis.get("score_data", {})
    num_freqs = sd.get("number_frequencies", {})
    all_numbers = sd.get("all_numbers", [])

    if num_freqs:
        num_data = [
            {
                "Value (canonical)": val,
                "Frequency": f"{count}/{n}",
                "Consistent": "✅" if count >= n / 2 else "⚠️",
            }
            for val, count in sorted(num_freqs.items(), key=lambda x: -x[1])
        ]
        st.dataframe(pd.DataFrame(num_data), use_container_width=True, hide_index=True)

        with st.expander("Numbers extracted per run (SI-normalized)"):
            for i, nums in enumerate(all_numbers, 1):
                st.markdown(f"**Run {i}:** `{', '.join(nums) if nums else 'none'}`")
    else:
        st.info("No numeric values with units extracted.")

st.markdown("---")

# =============================================================================
# SIDE-BY-SIDE COMPARISON
# =============================================================================
st.subheader("🔬 Compare Individual Runs")

col_a, col_b = st.columns(2)
with col_a:
    it1 = st.selectbox("Run A", range(1, n + 1), index=0, key="run_a")
with col_b:
    it2 = st.selectbox("Run B", range(1, n + 1), index=min(1, n - 1), key="run_b")

c1, c2 = st.columns(2)
c1.text_area(f"Run {it1} ({data[it1-1].get('latency_sec', '?')}s)", outputs[it1-1], height=300)
c2.text_area(f"Run {it2} ({data[it2-1].get('latency_sec', '?')}s)", outputs[it2-1], height=300)

if st.checkbox("Show similarity %", value=True):
    sim = SequenceMatcher(None, outputs[it1-1], outputs[it2-1]).ratio()
    st.info(f"Character-level similarity between Run {it1} and Run {it2}: **{sim:.1%}**")

st.markdown("---")

# =============================================================================
# EXPORT
# =============================================================================
st.subheader("📄 Export")

if analysis:
    try:
        from analyzer import generate_report
        report_md = generate_report(data, analysis)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Markdown Report",
                data=report_md,
                file_name="validation_report.md",
                mime="text/markdown",
            )
        with col2:
            st.download_button(
                "📥 Download Analysis JSON",
                data=json.dumps(analysis, indent=4),
                file_name="validation_analysis.json",
                mime="application/json",
            )
        with st.expander("Preview Report"):
            st.markdown(report_md)
    except Exception as e:
        st.error(f"Could not generate report: {e}")

st.markdown("---")
st.caption("AI Output Validation Framework v3.0 | Proof of concept — not production ready")
