"""
Review Insights — Streamlit Dashboard
Consumes the FastAPI backend over HTTP.
"""

import os
import streamlit as st
import requests
import pandas as pd
import altair as alt

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# Mixed-theme warning list
MIXED_THEMES = {5, 20, 28}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(endpoint: str, params: dict | None = None) -> requests.Response:
    """Make a GET request to the FastAPI backend. Raises on connection error."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp
    except requests.ConnectionError:
        st.error(
            f"⚠️ **Cannot reach the API at `{API_BASE_URL}`.**\n\n"
            f"Make sure the FastAPI server is running:\n\n"
            f"```\nuvicorn src.api.main:app --reload\n```"
        )
        st.stop()
    except requests.HTTPError as exc:
        st.error(f"⚠️ API returned an error: **{exc.response.status_code}** — {exc.response.text}")
        st.stop()


@st.cache_data(ttl=300, show_spinner="Fetching sentiment data…")
def fetch_sentiment() -> dict:
    return api_get("/sentiment").json()


@st.cache_data(ttl=300, show_spinner="Fetching themes…")
def fetch_themes() -> list[dict]:
    return api_get("/themes").json()


@st.cache_data(ttl=300, show_spinner="Fetching summary…")
def fetch_summary(theme_id: int) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE_URL}/summary/{theme_id}", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"⚠️ Cannot reach the API at `{API_BASE_URL}`.")
        st.stop()


@st.cache_data(ttl=60, show_spinner="Searching reviews…")
def fetch_search(query: str) -> list[dict]:
    return api_get("/search", params={"q": query}).json()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Review Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #8b93a7 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Review Insights Dashboard")
st.caption("Powered by FastAPI · 50,000 Amazon reviews · 29 themes + Unclustered")

tab_overview, tab_explorer, tab_search, tab_dropbox = st.tabs([
    "📈 Overview",
    "🔍 Theme Explorer",
    "🔎 Search",
    "Drop your data",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════════════════════════════════════════

with tab_overview:
    sentiment_data = fetch_sentiment()
    overall = sentiment_data["overall"]

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{overall['total_reviews']:,}")
    col2.metric("Positive", f"{overall['positive_count']:,}")
    col3.metric("Negative", f"{overall['negative_count']:,}")
    col4.metric("% Negative", f"{overall['pct_negative']}%")

    st.divider()

    # Info callout
    st.info(
        "📌 **45% of reviews are unclustered; themes cover the rest.** "
        "The unclustered group (topic −1) contains reviews that did not cluster "
        "into any of the 29 discovered themes."
    )

    # Checkbox to include Unclustered
    include_unclustered = st.checkbox(
        "Include Unclustered (topic −1) in chart",
        value=False,
        key="overview_unclustered",
    )

    # Build DataFrame for chart
    themes = sentiment_data["themes"]
    df_themes = pd.DataFrame(themes)

    if not include_unclustered:
        df_themes = df_themes[df_themes["topic_id"] != -1]

    df_themes = df_themes.sort_values("pct_negative", ascending=False)

    # Altair horizontal bar chart
    chart = (
        alt.Chart(df_themes, title="Negative Sentiment by Theme (%)")
        .mark_bar(
            cornerRadiusEnd=6,
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#3b82f6", offset=0),
                    alt.GradientStop(color="#ef4444", offset=1),
                ],
                x1=0, x2=1, y1=0, y2=0,
            ),
        )
        .encode(
            x=alt.X("pct_negative:Q", title="% Negative", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("label:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("label:N", title="Theme"),
                alt.Tooltip("count:Q", title="Reviews", format=","),
                alt.Tooltip("pct_negative:Q", title="% Negative", format=".1f"),
            ],
        )
        .properties(height=max(len(df_themes) * 28, 300))
    )

    st.altair_chart(chart, width="stretch")

    # Summary table
    with st.expander("📋 Full theme sentiment table"):
        display_df = df_themes[["topic_id", "label", "count", "positive_count", "negative_count", "pct_negative"]].copy()
        display_df.columns = ["Topic ID", "Theme", "Reviews", "Positive", "Negative", "% Negative"]
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Theme Explorer
# ═══════════════════════════════════════════════════════════════════════════

with tab_explorer:
    themes_list = fetch_themes()

    # Build display options: "ID: Label"
    theme_options = {
        t["topic_id"]: f"Theme {t['topic_id']}: {t['label']}"
        for t in themes_list
    }

    selected_id = st.selectbox(
        "Select a theme to explore",
        options=list(theme_options.keys()),
        format_func=lambda x: theme_options[x],
        index=1,  # Default to topic 0 (Books), skip -1
        key="explorer_theme",
    )

    # Find theme metadata
    theme_meta = next((t for t in themes_list if t["topic_id"] == selected_id), None)

    if theme_meta:
        # Metrics row
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Theme", theme_meta["label"])
        mc2.metric("Review Count", f"{theme_meta['count']:,}")
        mc3.metric("% Negative", f"{theme_meta['pct_negative']}%")

        # Mixed-theme caution
        if selected_id in MIXED_THEMES:
            st.warning(
                "⚠️ **Mixed theme: summary may blend product types.** "
                "This theme groups products from diverse subcategories, so the "
                "grounded summary may reference different product types together."
            )

        st.divider()

        # Fetch and display summary
        summary_data = fetch_summary(selected_id)

        if summary_data is None:
            st.info(
                "ℹ️ No grounded summary available for this theme. "
                "Unclustered reviews (topic −1) are not summarized by design."
            )
        else:
            # Summary paragraph
            st.subheader("📝 Summary")
            summary_text = summary_data.get("summary", "")
            if isinstance(summary_text, str):
                st.markdown(summary_text)
            elif isinstance(summary_text, dict):
                st.markdown(summary_text.get("text", ""))

            # Praises
            praises = summary_data.get("praises", [])
            if praises:
                st.subheader("👍 Praises")
                for i, p in enumerate(praises):
                    text = p.get("text", p) if isinstance(p, dict) else str(p)
                    ids = p.get("review_ids", []) if isinstance(p, dict) else []
                    id_str = ", ".join(str(rid) for rid in ids) if ids else "—"
                    st.markdown(f"**{i+1}.** {text}")
                    st.caption(f"Cited reviews: {id_str}")

            # Complaints
            complaints = summary_data.get("complaints", [])
            if complaints:
                st.subheader("👎 Complaints")
                for i, c in enumerate(complaints):
                    text = c.get("text", c) if isinstance(c, dict) else str(c)
                    ids = c.get("review_ids", []) if isinstance(c, dict) else []
                    id_str = ", ".join(str(rid) for rid in ids) if ids else "—"
                    st.markdown(f"**{i+1}.** {text}")
                    st.caption(f"Cited reviews: {id_str}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Search
# ═══════════════════════════════════════════════════════════════════════════

with tab_search:
    st.subheader("🔎 Semantic Review Search")
    st.caption(
        "Enter a natural-language query. The API encodes it with "
        "`all-MiniLM-L6-v2` and returns the 10 most similar reviews by cosine similarity."
    )

    query = st.text_input(
        "Search query",
        placeholder="e.g. battery died quickly, great jazz album, coffee maker leaking...",
        key="search_query",
    )

    if query and query.strip():
        results = fetch_search(query.strip())

        if not results:
            st.info("No results found.")
        else:
            st.success(f"**{len(results)} results** for: _{query.strip()}_")

            for i, r in enumerate(results):
                sentiment_emoji = "🟢" if r["sentiment_true"] == "positive" else "🔴"
                with st.container():
                    st.markdown(
                        f"**#{i+1}** &nbsp; {sentiment_emoji} &nbsp; "
                        f"Review `{r['review_id']}` &nbsp;·&nbsp; "
                        f"**{r['topic_label']}** &nbsp;·&nbsp; "
                        f"Score: `{r['similarity_score']:.4f}`"
                    )
                    st.markdown(f"> {r['text']}")
                    st.divider()
    elif query is not None and query.strip() == "":
        # User cleared the input — do nothing
        pass


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Drop your data
# ═══════════════════════════════════════════════════════════════════════════

with tab_dropbox:
    st.subheader("📥 Drop Your Data")
    st.caption("🔒 Processed in memory by this app's API, not stored, no LLM used.")

    # Limits display
    try:
        lim_resp = requests.get(f"{API_BASE_URL}/dropbox/limits", timeout=5)
        if lim_resp.status_code == 200:
            limits_info = lim_resp.json()
            max_mb = limits_info.get("max_file_size_mb", 5)
        else:
            max_mb = 5
    except Exception:
        max_mb = 5

    st.markdown(
        f"**Limits:** Max {max_mb} MB per file. CSV files over 3,000 rows are randomly sampled; "
        "Excel files over 3,000 rows are rejected."
    )

    # API Warnings
    st.warning(
        "⚠️ **Important Considerations:**\n"
        "- Models were trained on Amazon product reviews; accuracy on other text may be lower.\n"
        "- Themes are the 29 Amazon-product themes found in this project; they may not fit your data."
    )

    # Controls
    col_up, col_opt = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx"],
            key="dropbox_file_uploader",
        )
    with col_opt:
        text_column_input = st.text_input(
            "Text column name (optional)",
            placeholder="e.g. review, text, comment",
            key="dropbox_text_column",
            help="Leave blank to automatically detect the review text column.",
        )
        include_themes_check = st.checkbox(
            "Match reviews to the 29 themes",
            value=True,
            key="dropbox_match_themes",
        )

    # Action row: Analyze button + Clear results button
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        analyze_btn = st.button("Analyze Reviews", type="primary", key="dropbox_analyze_btn")
    with col_btn2:
        if "dropbox_results" in st.session_state and st.session_state["dropbox_results"] is not None:
            if st.button("Clear results", key="dropbox_clear_btn"):
                del st.session_state["dropbox_results"]
                st.rerun()

    # Handle Analyze click
    if analyze_btn:
        if uploaded_file is None:
            st.error("⚠️ Please select and upload a CSV or XLSX file first.")
        else:
            with st.spinner("Analyzing reviews… This may take a moment for large files."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    }
                    data_payload = {
                        "include_themes": "true" if include_themes_check else "false",
                    }
                    if text_column_input and text_column_input.strip():
                        data_payload["text_column"] = text_column_input.strip()

                    resp = requests.post(
                        f"{API_BASE_URL}/dropbox/analyze",
                        files=files,
                        data=data_payload,
                        timeout=300,
                    )

                    if resp.status_code == 200:
                        st.session_state["dropbox_results"] = resp.json()
                    elif resp.status_code == 400:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"⚠️ **Bad Request (400):** {detail}")
                    elif resp.status_code == 413:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"⚠️ **File Too Large (413):** {detail}")
                    elif resp.status_code == 415:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"⚠️ **Unsupported Media Type (415):** {detail}")
                    elif resp.status_code == 500:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"⚠️ **Internal Server Error (500):** {detail}")
                    else:
                        st.error(f"⚠️ API returned an error: **{resp.status_code}** — {resp.text}")
                except requests.ConnectionError:
                    st.error(
                        f"⚠️ **Cannot reach the API at `{API_BASE_URL}`.**\n\n"
                        f"Make sure the FastAPI server is running:\n\n"
                        f"```\nuvicorn src.api.main:app --reload\n```"
                    )
                except requests.Timeout:
                    st.error("⚠️ The request timed out after 300 seconds.")
                except Exception as e:
                    st.error(f"⚠️ An unexpected error occurred: {str(e)}")

    # Display Results if present
    if "dropbox_results" in st.session_state and st.session_state["dropbox_results"] is not None:
        res = st.session_state["dropbox_results"]
        rep = res.get("report", {})
        summ = res.get("summary", {})

        st.divider()
        st.subheader("📊 Analysis Results")

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows Used", f"{rep.get('rows_used', 0):,}")
        total_removed = (
            rep.get("removed_empty", 0)
            + rep.get("removed_under_3_words", 0)
            + rep.get("removed_duplicates", 0)
        )
        k2.metric("Rows Removed", f"{total_removed:,}")
        k3.metric("% Positive", f"{summ.get('pct_positive', 0)}%")
        k4.metric("% Negative", f"{summ.get('pct_negative', 0)}%")

        # Sub-KPI details
        st.caption(
            f"**Data breakdown:** {rep.get('rows_in', 0):,} total input rows · "
            f"{rep.get('removed_empty', 0):,} empty removed · "
            f"{rep.get('removed_under_3_words', 0):,} under 3 words removed · "
            f"{rep.get('removed_duplicates', 0):,} duplicates removed"
            + (" · *(Randomly sampled to 3,000 rows)*" if rep.get("sampled") else "")
        )

        # Low confidence notice under KPIs
        st.info("The model has no neutral class; scores near 0.5 mean the text has no clear sentiment.")

        # Accuracy and Macro F1 vs Ratings
        acc_info = res.get("accuracy_vs_ratings")
        if acc_info is not None:
            st.markdown("### 🎯 Ground-Truth Rating Comparison")
            st.caption("Ratings 1–2 treated as negative, 4–5 as positive; 3-star rows are ignored.")
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Accuracy", f"{acc_info.get('accuracy', 0):.3f}")
            ac2.metric("Macro F1", f"{acc_info.get('macro_f1', 0):.3f}")
            ac3.metric("Evaluated Rows", f"{acc_info.get('evaluated_count', 0):,}")
            ac4.metric("3-Star Rows Ignored", f"{acc_info.get('ignored_rating_3_count', 0):,}")

        # Discovered Themes
        themes_data = res.get("themes_table", [])
        if themes_data:
            st.markdown("### 🏷️ Theme Breakdown")
            df_th = pd.DataFrame(themes_data)

            # Horizontal bar chart of rows per theme
            th_chart = (
                alt.Chart(df_th, title="Rows per Theme")
                .mark_bar(
                    cornerRadiusEnd=6,
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="#3b82f6", offset=0),
                            alt.GradientStop(color="#8b5cf6", offset=1),
                        ],
                        x1=0, x2=1, y1=0, y2=0,
                    ),
                )
                .encode(
                    x=alt.X("count:Q", title="Number of Rows"),
                    y=alt.Y("label:N", title=None, sort="-x"),
                    tooltip=[
                        alt.Tooltip("label:N", title="Theme"),
                        alt.Tooltip("count:Q", title="Rows", format=","),
                        alt.Tooltip("share:Q", title="Share (%)", format=".1f"),
                        alt.Tooltip("pct_negative:Q", title="% Negative", format=".1f"),
                    ],
                )
                .properties(height=max(len(df_th) * 26, 260))
            )
            st.altair_chart(th_chart, width="stretch")

            # Theme Table
            df_th_disp = df_th[["label", "count", "share", "pct_negative"]].copy()
            df_th_disp.columns = ["Theme", "Rows", "Share (%)", "% Negative"]
            st.dataframe(df_th_disp, width="stretch", hide_index=True)

        # 20-row preview
        preview_data = res.get("preview", [])
        if preview_data:
            st.markdown("### 👁️ Preview (First 20 Rows)")
            df_prev = pd.DataFrame(preview_data)

            # Mark low confidence (< 0.65)
            df_prev["confidence"] = df_prev["probability"].apply(
                lambda p: f"{p:.4f} (low confidence)" if p < 0.65 else f"{p:.4f}"
            )

            cols_show = ["row_id", "scrubbed_text", "label", "confidence"]
            col_headers = ["Row", "Text", "Sentiment", "Probability"]
            if "theme" in df_prev.columns:
                cols_show.append("theme")
                col_headers.append("Theme")

            df_prev_disp = df_prev[cols_show].copy()
            df_prev_disp.columns = col_headers
            st.dataframe(df_prev_disp, width="stretch", hide_index=True)

        # Download button for scored CSV
        full_rows_data = res.get("rows", [])
        if full_rows_data:
            df_download = pd.DataFrame(full_rows_data)
            csv_bytes = df_download.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored CSV",
                data=csv_bytes,
                file_name="scored_reviews.csv",
                mime="text/csv",
                key="dropbox_download_csv_btn",
            )

