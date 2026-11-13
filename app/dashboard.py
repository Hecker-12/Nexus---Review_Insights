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


@st.cache_data(ttl=300, show_spinner="Fetching methodology data…")
def fetch_methodology() -> dict:
    return api_get("/methodology").json()


@st.cache_data(ttl=300, show_spinner="Fetching review…")
def fetch_review(review_id: int) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE_URL}/review/{review_id}", timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"⚠️ Cannot reach the API at `{API_BASE_URL}`.")
        st.stop()
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner="Fetching similar reviews…")
def fetch_similar(review_id: int, k: int = 5) -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE_URL}/similar/{review_id}", params={"k": k}, timeout=10)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"⚠️ Cannot reach the API at `{API_BASE_URL}`.")
        st.stop()
    except Exception:
        return []


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

tab_overview, tab_explorer, tab_search, tab_dropbox, tab_methodology, tab_complaints = st.tabs([
    "📈 Overview",
    "🔍 Theme Explorer",
    "🔎 Search",
    "Drop your data",
    "Methodology",
    "Complaints (exploratory)",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════════════════════════════════════════

with tab_overview:
    sentiment_data = fetch_sentiment()
    overall = sentiment_data["overall"]
    themes_data = sentiment_data.get("themes", [])

    # Key insights card computed from /sentiment data
    themes_ge_100 = [t for t in themes_data if t["topic_id"] != -1 and t["count"] >= 100]
    top3_neg = sorted(themes_ge_100, key=lambda t: t["pct_negative"], reverse=True)[:3]
    unclustered_t = next((t for t in themes_data if t["topic_id"] == -1), {"count": 22623})
    unclustered_share = round((unclustered_t["count"] / overall["total_reviews"]) * 100.0, 1) if overall["total_reviews"] > 0 else 45.2

    with st.container():
        st.subheader("💡 Key Insights")
        ki_col1, ki_col2 = st.columns([3, 2])
        with ki_col1:
            st.markdown(
                f"- **Overall Negative Sentiment**: **{overall['pct_negative']}%** of the {overall['total_reviews']:,} customer reviews are negative.\n"
                f"- **Unclustered Reviews**: **{unclustered_share}%** of reviews do not cluster into any specific product theme.\n"
                f"- **Top 3 Most Negative Themes (≥ 100 reviews)**:\n"
                f"  1. **{top3_neg[0]['label']}**: {top3_neg[0]['pct_negative']}% negative ({top3_neg[0]['count']:,} reviews)\n"
                f"  2. **{top3_neg[1]['label']}**: {top3_neg[1]['pct_negative']}% negative ({top3_neg[1]['count']:,} reviews)\n"
                f"  3. **{top3_neg[2]['label']}**: {top3_neg[2]['pct_negative']}% negative ({top3_neg[2]['count']:,} reviews)"
            )
        with ki_col2:
            donut_df = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Count": [overall["positive_count"], overall["negative_count"]],
            })
            donut_chart = (
                alt.Chart(donut_df)
                .mark_arc(innerRadius=45, outerRadius=80)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color(
                        "Sentiment:N",
                        scale=alt.Scale(domain=["Positive", "Negative"], range=["#22c55e", "#ef4444"]),
                        legend=alt.Legend(title="Sentiment", orient="right"),
                    ),
                    tooltip=[alt.Tooltip("Sentiment:N"), alt.Tooltip("Count:Q", format=",d")],
                )
                .properties(width=190, height=190)
            )
            st.altair_chart(donut_chart, width="content")

    st.divider()

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

            # Open a cited review
            all_cited_ids = []
            for p in praises:
                if isinstance(p, dict):
                    all_cited_ids.extend(p.get("review_ids", []))
            for c in complaints:
                if isinstance(c, dict):
                    all_cited_ids.extend(c.get("review_ids", []))
            for cl in summary_data.get("claims", []):
                if isinstance(cl, dict):
                    all_cited_ids.extend(cl.get("review_ids", []))

            unique_cited_ids = sorted({int(rid) for rid in all_cited_ids if str(rid).isdigit()})
            if unique_cited_ids:
                st.divider()
                st.subheader("📖 Open a cited review")
                chosen_rid = st.selectbox(
                    "Open a cited review",
                    options=unique_cited_ids,
                    key=f"cited_review_{selected_id}",
                )
                if chosen_rid is not None:
                    r_info = fetch_review(chosen_rid)
                    if r_info:
                        s_emoji = "🟢" if r_info["sentiment_true"] == "positive" else "🔴"
                        st.markdown(
                            f"**Review `{r_info['review_id']}`** &nbsp;·&nbsp; {s_emoji} "
                            f"**{r_info['sentiment_true'].capitalize()}** &nbsp;·&nbsp; Theme: **{r_info['theme_label']}**"
                        )
                        st.info(r_info["text"])


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

            # Reviews about the same subject
            result_ids = [r["review_id"] for r in results]
            if result_ids:
                st.subheader("🎯 Reviews about the same subject")
                st.caption("Similarity is topical, not about sentiment.")
                chosen_sim_id = st.selectbox(
                    "Find reviews similar to",
                    options=result_ids,
                    key="search_sim_select",
                )
                if chosen_sim_id is not None:
                    sim_items = fetch_similar(chosen_sim_id, k=5)
                    if sim_items:
                        for s_idx, s_item in enumerate(sim_items):
                            s_emoji = "🟢" if s_item["sentiment_true"] == "positive" else "🔴"
                            with st.container():
                                st.markdown(
                                    f"**#{s_idx+1}** &nbsp; {s_emoji} &nbsp; Review `{s_item['review_id']}` &nbsp;·&nbsp; "
                                    f"**{s_item['theme_label']}** &nbsp;·&nbsp; Sentiment: **{s_item['sentiment_true'].capitalize()}** &nbsp;·&nbsp; "
                                    f"Score: `{s_item['score']:.4f}`"
                                )
                                st.markdown(f"> {s_item['text']}")
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


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Methodology
# ═══════════════════════════════════════════════════════════════════════════

with tab_methodology:
    st.subheader("🔬 Methodology & Evaluation Rigor")
    st.caption("Comprehensive statistical evaluation on 10,000 held-out test reviews from `test.parquet`.")

    meth_data = fetch_methodology()

    # 1. Model Performance with 95% Bootstrap CIs
    st.markdown("### 📊 Model Performance (95% Bootstrap CIs, 2,000 resamples)")
    st.caption("Zero-shot = off-the-shelf SST-2 sentiment model, not trained on Amazon data.")
    eval_table = meth_data.get("evaluation_upgrade.csv")
    if eval_table:
        df_eval = pd.DataFrame(eval_table["rows"], columns=eval_table["columns"])
        models_order = [
            "VADER default",
            "VADER tuned (0.7)",
            "TF-IDF+LogReg",
            "DistilBERT zero-shot",
            "DistilBERT fine-tuned",
            "Ensemble (TF-IDF + FT)",
        ]

        summary_rows = []
        for m in models_order:
            m_rows = df_eval[df_eval["model"] == m]
            if not m_rows.empty:
                acc_row = m_rows[m_rows["metric"] == "accuracy"]
                f1_row = m_rows[m_rows["metric"] == "macro_f1"]
                acc_val = acc_row["value"].values[0] if not acc_row.empty else 0.0
                acc_l = acc_row["ci_low"].values[0] if not acc_row.empty else 0.0
                acc_h = acc_row["ci_high"].values[0] if not acc_row.empty else 0.0

                f1_val = f1_row["value"].values[0] if not f1_row.empty else 0.0
                f1_l = f1_row["ci_low"].values[0] if not f1_row.empty else 0.0
                f1_h = f1_row["ci_high"].values[0] if not f1_row.empty else 0.0

                summary_rows.append({
                    "Model": m,
                    "Accuracy (95% CI)": f"{acc_val:.4f} ({acc_l:.4f} to {acc_h:.4f})",
                    "Macro F1 (95% CI)": f"{f1_val:.4f} ({f1_l:.4f} to {f1_h:.4f})",
                })

        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        # Paired differences
        st.markdown("#### Paired Accuracy Differences (95% Bootstrap CI)")
        diff_rows = df_eval[df_eval["metric"] == "accuracy_diff"]
        if not diff_rows.empty:
            diff_disp = []
            for _, r in diff_rows.iterrows():
                ci_l = float(r["ci_low"])
                ci_h = float(r["ci_high"])
                diff_val = float(r["value"])
                excludes_zero = (ci_l > 0 and ci_h > 0) or (ci_l < 0 and ci_h < 0)
                conclusion = "Difference detected (95% CI excludes 0)" if excludes_zero else "No detectable difference (95% CI includes 0)"
                diff_disp.append({
                    "Comparison": r["model"],
                    "Accuracy Difference": f"{diff_val:+.4f} ({ci_l:.4f} to {ci_h:.4f})",
                    "Statistical Conclusion": conclusion,
                })
            st.dataframe(pd.DataFrame(diff_disp), width="stretch", hide_index=True)
    else:
        st.info("not available in this deployment")

    st.divider()

    # 2. McNemar Exact Discordant Tests
    st.markdown("### ⚖️ McNemar Exact Tests (Discordant Pairs)")
    mcnemar_table = meth_data.get("mcnemar_tests.csv")
    if mcnemar_table:
        df_mcn = pd.DataFrame(mcnemar_table["rows"], columns=mcnemar_table["columns"])
        df_mcn_disp = []
        for _, r in df_mcn.iterrows():
            p_val = float(r["p_value"])
            conclusion = "Difference detected (p < 0.001)" if p_val < 0.001 else "No detectable difference (p >= 0.001)"
            df_mcn_disp.append({
                "Comparison": r["comparison"],
                "Model 1 Right, Model 2 Wrong (b)": int(r["b"]),
                "Model 1 Wrong, Model 2 Right (c)": int(r["c"]),
                "Total Discordant (b + c)": int(r["total_discordant"]),
                "Two-Sided p-value": f"{p_val:.6e}",
                "Conclusion": conclusion,
            })
        st.dataframe(pd.DataFrame(df_mcn_disp), width="stretch", hide_index=True)
    else:
        st.info("not available in this deployment")

    st.divider()

    # 3. Confidence Calibration & ECE
    st.markdown("### 🎯 Confidence Calibration & Expected Calibration Error")
    calib_table = meth_data.get("calibration.csv")
    calib_summary_table = meth_data.get("calibration_summary.csv")

    if calib_table:
        df_calib = pd.DataFrame(calib_table["rows"], columns=calib_table["columns"])
        c_col1, c_col2 = st.columns([3, 2])

        with c_col1:
            valid_calib = df_calib.dropna(subset=["accuracy"])
            lines_chart = (
                alt.Chart(valid_calib)
                .mark_line(point=True)
                .encode(
                    x=alt.X("mean_confidence:Q", title="Mean Predicted Confidence", scale=alt.Scale(domain=[0.48, 1.0])),
                    y=alt.Y("accuracy:Q", title="Empirical Accuracy", scale=alt.Scale(domain=[0.4, 1.02])),
                    color=alt.Color("model:N", title="Model"),
                    tooltip=["model:N", "bin_index:Q", alt.Tooltip("mean_confidence:Q", format=".4f"), alt.Tooltip("accuracy:Q", format=".4f"), "count:Q"],
                )
            )

            diag_data = pd.DataFrame({"x": [0.5, 1.0], "y": [0.5, 1.0]})
            diag_chart = (
                alt.Chart(diag_data)
                .mark_line(strokeDash=[4, 4], color="#64748b")
                .encode(x="x:Q", y="y:Q")
            )

            chart_calib = (lines_chart + diag_chart).properties(title="Reliability Diagram", height=320)
            st.altair_chart(chart_calib, width="stretch")

        with c_col2:
            st.markdown("#### ECE Summary")
            if calib_summary_table:
                df_cs = pd.DataFrame(calib_summary_table["rows"], columns=calib_summary_table["columns"])
                df_cs_disp = df_cs.copy()
                df_cs_disp.columns = ["Model", "ECE"]
                df_cs_disp["ECE"] = df_cs_disp["ECE"].apply(lambda v: f"{v:.4f}")
                st.dataframe(df_cs_disp, width="stretch", hide_index=True)

            st.markdown("#### Fine-Tuned Accuracy by Confidence Band")
            slice_table_for_conf = meth_data.get("slice_analysis.csv")
            if slice_table_for_conf:
                df_sl_raw = pd.DataFrame(slice_table_for_conf["rows"], columns=slice_table_for_conf["columns"])
                ft_bands = df_sl_raw[df_sl_raw["slice_category"] == "Fine-Tuned Confidence Bands"]
                if not ft_bands.empty:
                    ft_bands_disp = ft_bands[["slice_name", "n", "acc_fine_tuned"]].copy()
                    ft_bands_disp.columns = ["Band", "Reviews (n)", "Accuracy"]
                    ft_bands_disp["Accuracy"] = ft_bands_disp["Accuracy"].apply(lambda a: f"{a:.4f}")
                    st.dataframe(ft_bands_disp, width="stretch", hide_index=True)
                    st.caption(
                        "Excluding the 639 reviews with fine-tuned confidence below 0.90, accuracy on the remaining 9,361 reviews is 0.9686. "
                        "The band edges were fixed in advance; this is descriptive and not a tuned threshold."
                    )
    else:
        st.info("not available in this deployment")

    st.divider()

    # 4. Slice Analysis Table
    st.markdown("### 🍰 Slice Analysis")
    slice_table = meth_data.get("slice_analysis.csv")
    if slice_table:
        df_slice = pd.DataFrame(slice_table["rows"], columns=slice_table["columns"])
        df_slice_disp = df_slice.copy()
        df_slice_disp["small_n"] = df_slice_disp["small_n"].apply(lambda s: "⚠️ Small n (<200)" if s else "No")
        df_slice_disp["acc_tfidf"] = df_slice_disp["acc_tfidf"].apply(lambda x: f"{x:.4f}")
        df_slice_disp["acc_zero_shot"] = df_slice_disp["acc_zero_shot"].apply(lambda x: f"{x:.4f}")
        df_slice_disp["acc_fine_tuned"] = df_slice_disp["acc_fine_tuned"].apply(lambda x: f"{x:.4f}")
        df_slice_disp.columns = ["Category", "Slice", "n", "Small n Flag", "TF-IDF Acc", "Zero-Shot Acc", "Fine-Tuned Acc"]
        st.dataframe(df_slice_disp, width="stretch", hide_index=True)
    else:
        st.info("not available in this deployment")

    st.divider()

    # 5. Topic Modeling Comparison Table
    st.markdown("### 🗂️ Topic Modeling Comparison")
    topic_table = meth_data.get("topic_comparison.csv")
    if topic_table:
        df_topic = pd.DataFrame(topic_table["rows"], columns=topic_table["columns"])
        st.dataframe(df_topic, width="stretch", hide_index=True)
        st.caption("📌 **Note**: BERTopic NPMI excludes the outlier group.")
    else:
        st.info("not available in this deployment")

    st.divider()

    # 6. Methodological Limits (Verbatim from results/evaluation_upgrade.md)
    st.markdown("### ⚠️ Methodological Limits")
    st.caption("*(Embedded verbatim from `results/evaluation_upgrade.md` Section 5)*")
    st.markdown(
        "- **Sampling variability**: Confidence intervals reflect test set sampling variability only and do not account for data collection bias or annotation shifts.\n"
        "- **Training randomness**: each model was trained once, so the intervals do not include variation from retraining.\n"
        "- **Confidence calibration**: Calibration is evaluated against binary correctness using the maximum predicted class probability.\n"
        "- **Sample size in slices**: Several slices have sample size below 200 reviews (`small_n = True`), increasing variance in slice-level estimates.\n"
        "- **Untuned ensemble**: The ensemble model combines TF-IDF and fine-tuned transformer probabilities using fixed equal weights (0.5 / 0.5) without parameter tuning.\n"
        "- **Single test set**: All evaluations are conducted on a single held-out 10,000-review test split."
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: Complaints (exploratory)
# ═══════════════════════════════════════════════════════════════════════════

with tab_complaints:
    st.subheader("⚠️ Complaint Taxonomy Analysis")
    st.info(
        "ℹ️ **Exploratory**: a predefined taxonomy matched by similarity; shares are lower bounds because "
        "about 48% of negative reviews remain unmatched; one type is unreliable."
    )

    meth_data = fetch_methodology()
    sentiment_info = fetch_sentiment()
    themes_info = sentiment_info.get("themes", [])

    # 1. Bar Chart of Complaint Types
    counts_table = meth_data.get("complaint_type_counts_v2.csv")
    if counts_table:
        st.markdown("### 📊 Overall Distribution by Complaint Type")
        df_counts = pd.DataFrame(counts_table["rows"], columns=counts_table["columns"])
        df_counts["complaint_type"] = df_counts["complaint_type"].replace({"Too loud or noisy": "Too loud or noisy (unreliable)"})

        bar_chart = (
            alt.Chart(df_counts)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("count:Q", title="Number of Negative Reviews"),
                y=alt.Y("complaint_type:N", title=None, sort="-x"),
                tooltip=[
                    alt.Tooltip("complaint_type:N", title="Complaint Type"),
                    alt.Tooltip("count:Q", title="Reviews", format=",d"),
                    alt.Tooltip("share:N", title="Share"),
                ],
                color=alt.condition(
                    alt.datum.complaint_type == "Other / unclear",
                    alt.value("#64748b"),
                    alt.value("#f97316"),
                ),
            )
            .properties(height=max(len(df_counts) * 26, 260))
        )
        st.altair_chart(bar_chart, width="stretch")
    else:
        st.info("Complaint counts table not available in this deployment.")

    st.divider()

    # 2. Heatmap: Complaint Type by Product Theme (themes with >= 100 negative reviews)
    st.markdown("### 🔥 Complaint Type by Product Theme (≥ 100 Negative Reviews)")
    theme_table = meth_data.get("complaint_by_theme_v2.csv")

    if theme_table:
        df_th_complaints = pd.DataFrame(theme_table["rows"], columns=theme_table["columns"])

        # Filter to themes with negative_count >= 100 from /sentiment
        qualifying_labels = {t["label"] for t in themes_info if t.get("negative_count", 0) >= 100}
        df_th_filtered = df_th_complaints[df_th_complaints["product_theme"].isin(qualifying_labels)].copy()

        if "Too loud or noisy" in df_th_filtered.columns:
            df_th_filtered = df_th_filtered.rename(columns={"Too loud or noisy": "Too loud or noisy (unreliable)"})

        n_named_themes = len([t for t in df_th_filtered["product_theme"] if t != "Unclustered"])
        has_unclustered = "Unclustered" in df_th_filtered["product_theme"].values
        theme_desc = f"{n_named_themes} product themes plus the Unclustered group" if has_unclustered else f"{len(df_th_filtered)} product themes"
        st.caption(
            f"Displaying {theme_desc}, each with at least 100 negative reviews. "
            "'Other / unclear' is excluded from heatmap colors to highlight specific complaint variations."
        )

        value_cols = [c for c in df_th_filtered.columns if c not in ["product_theme", "Other / unclear"]]
        df_melted = df_th_filtered.melt(
            id_vars=["product_theme"],
            value_vars=value_cols,
            var_name="complaint_type",
            value_name="percentage",
        )

        heatmap_chart = (
            alt.Chart(df_melted)
            .mark_rect()
            .encode(
                x=alt.X("complaint_type:N", title="Complaint Type"),
                y=alt.Y("product_theme:N", title="Product Theme", sort="y"),
                color=alt.Color("percentage:Q", title="% of Negatives", scale=alt.Scale(scheme="reds")),
                tooltip=[
                    alt.Tooltip("product_theme:N", title="Theme"),
                    alt.Tooltip("complaint_type:N", title="Complaint"),
                    alt.Tooltip("percentage:Q", title="Percentage", format=".2f"),
                ],
            )
            .properties(height=max(len(df_th_filtered) * 22, 400))
        )
        st.altair_chart(heatmap_chart, width="stretch")

        with st.expander("📋 View full theme complaint breakdown table"):
            st.dataframe(df_th_filtered, width="stretch", hide_index=True)
    else:
        st.info("Theme complaint breakdown table not available in this deployment.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "**Models used:** VADER (rule-based), TF-IDF + Logistic Regression, DistilBERT (zero-shot & fine-tuned), "
    "all-MiniLM-L6-v2 (embeddings), BERTopic (topic discovery) · "
    "⚠️ **Domain warning:** Models were trained on Amazon product reviews; accuracy on other text may be lower · "
    "🔍 Reviews of claims and topics were done by an LLM reviewer, not a human."
)

