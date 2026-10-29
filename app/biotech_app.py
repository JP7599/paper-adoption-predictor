"""Paper Adoption Predictor, biotech. Run with:  .venv/bin/streamlit run app/biotech_app.py

Pages: the ranked feed of new bioRxiv preprints (a switch takes the authors out of the equation), with a plain-words search, filters,
the top five and the hidden gems, and per paper the verdict, the reasons and a link; score your own paper; rising topics; how it works.
Model and tests: notebooks 06-14, src/paper_adoption/predict.py.
Look: the theme in .streamlit/config.toml and the CSS below. White page, black header band, dark green details, Fira Sans + Zilla Slab.
"""
import html
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parents[1] / "src"           # on Hugging Face Spaces the package is not installed, it sits in the repo
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from paper_adoption import ibcs  # noqa: E402
from paper_adoption.predict import CATEGORIES, EXPECTED, POOL_DIR, Scorer, build_pool, load_pool, names_from_biorxiv, search, verdict  # noqa: E402

ASSETS = Path(__file__).parent / "assets"
if not (ASSETS / "favicon.png").exists():                    # first run on a new machine: draw the small hexagon favicon
    import make_assets
    make_assets.favicon("favicon.png")

st.set_page_config(page_title="Paper Adoption Predictor", page_icon=str(ASSETS / "favicon.png"), layout="wide", initial_sidebar_state="expanded")
ibcs.style()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@500;600;700&family=Fira+Sans:wght@400;500;600&display=swap');
:root {--black:#141615; --green:#1f4d3a; --green-light:#8fc3a8; --line:#d9ddd9; --soft:#f4f5f4; --ink:#141615; --muted:#5c625e; --ochre:#a67c1e;
       --serif:"Zilla Slab", Georgia, "Times New Roman", serif; --sans:"Fira Sans", Helvetica, Arial, sans-serif;}

header[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1320px; padding-top: 3.2rem; padding-bottom: 4rem;}

/* sidebar */
.side-brand {display: flex; align-items: center; gap: 10px; color: var(--ink); font-family: var(--serif); font-size: 1.15rem; font-weight: 600; margin: 4px 0 16px 0;}
.side-brand .small {font-family: var(--sans); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--green); margin-left: 2px;}
.side-lbl {font-size: 0.74rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--green); margin: 8px 0 2px 0;}
.side-foot {font-size: 0.8rem; color: var(--muted); margin-top: 18px; line-height: 1.5;}

/* the header band: near black, white title, green details */
.hero {background: var(--black); color: #f3f4f3; padding: 28px 34px 24px 34px; border-radius: 6px; margin: 0 0 18px 0; border-top: 4px solid var(--green);}
.hero .kicker {font-size: 0.74rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--green-light); margin-bottom: 10px;}
.hero h1 {font-family: var(--serif); font-size: 2.5rem; font-weight: 600; line-height: 1.08; margin: 0 0 10px 0; padding: 0; color: #ffffff;}
.hero p {max-width: 56rem; font-size: 1.02rem; line-height: 1.5; color: #c9cfcb; margin: 0 0 16px 0;}
.hero .meta {font-size: 0.86rem; color: #a9b0ab; display: flex; flex-wrap: wrap; gap: 8px 26px; align-items: center;}
.hero .legend .pill {margin-right: 3px; font-size: 0.8rem; padding: 1px 8px;}

/* navigation: a segmented bar, the open page in white with green text */
div[role="tablist"] {gap: 4px; background: var(--soft); border: 1px solid var(--line); border-radius: 5px; padding: 4px; margin: 0 0 20px 0;}
div[data-testid="stTab"] {padding: 7px 16px; border-radius: 3px; height: auto; min-height: 0; border: 1px solid transparent;}
div[data-testid="stTab"]:hover {background: #e9ece9;}
div[data-testid="stTab"][aria-selected="true"] {background: #ffffff; border-color: var(--line);}
div[data-testid="stTab"] p {font-family: var(--sans); font-size: 0.97rem; font-weight: 500; color: var(--muted); letter-spacing: 0; text-transform: none;}
div[data-testid="stTab"][aria-selected="true"] p {color: var(--green); font-weight: 600;}
div[data-testid="stTab"] .react-aria-SelectionIndicator {display: none;}

/* section labels, text */
.section {display: flex; align-items: baseline; gap: 14px; margin: 24px 0 10px 0; padding-bottom: 6px; border-bottom: 1px solid var(--line);}
.section .lbl {font-size: 0.8rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--green); white-space: nowrap;}
.section .note {font-size: 0.88rem; color: var(--muted);}
.lead {color: var(--muted); font-size: 0.97rem; margin: 0 0 12px 0; max-width: 70rem;}
.muted {color: var(--muted); font-size: 0.9rem;}

/* pills and tags */
.pill {display: inline-block; padding: 2px 10px; border-radius: 3px; font-weight: 600; font-size: 0.92rem; white-space: nowrap; border: 1px solid transparent; font-variant-numeric: tabular-nums;}
.pill-high {background: var(--green); color: #ffffff;}
.pill-mid {background: #ffffff; color: var(--green); border-color: var(--green);}
.pill-low {background: #ffffff; color: var(--muted); border-color: #c9cdc9;}
.tag {display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 0.7rem; letter-spacing: 0.06em; text-transform: uppercase; background: var(--soft); color: var(--muted); border: 1px solid var(--line); vertical-align: 1px; white-space: nowrap;}
.tag-gem {background: #faf3e1; color: #7a5a10; border-color: #e6d6ab;}

/* cards: the top five, the hidden gems */
.card {position: relative; border: 1px solid var(--line); border-top: 3px solid var(--green); border-radius: 4px; padding: 14px 18px; background: #ffffff; margin-bottom: 12px; min-height: 200px;}
.card.gem {border-top-color: var(--ochre);}
.card .meta {color: var(--muted); font-size: 0.78rem; margin-bottom: 8px; padding-right: 26px; line-height: 1.6;}
.card h3 {font-family: var(--serif); margin: 0 0 10px 0; padding: 0; font-size: 1.12rem; line-height: 1.3; font-weight: 600; color: var(--ink); display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;}
.card .rank {position: absolute; top: 8px; right: 14px; font-family: var(--serif); font-size: 1.5rem; color: #b9bdb9; font-weight: 600; line-height: 1;}
.card .score {display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; color: var(--muted); font-size: 0.86rem;}
a.read {color: var(--green) !important; text-decoration: none; font-weight: 500; font-size: 0.88rem; border-bottom: 1px solid #b9cfc2;}
a.read:hover {border-bottom-color: var(--green);}

/* the selected paper */
.detail {border: 1px solid var(--line); border-left: 4px solid var(--green); border-radius: 4px; padding: 20px 26px; background: #ffffff; margin: 14px 0 10px 0;}
.detail .meta {color: var(--muted); font-size: 0.84rem;}
.detail h2 {font-family: var(--serif); font-size: 1.55rem; line-height: 1.25; margin: 6px 0 14px 0; padding: 0; color: var(--ink); font-weight: 600;}
.detail .scores {display: flex; gap: 36px; flex-wrap: wrap; margin: 4px 0 12px 0;}
.detail .lbl {display: block; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 4px;}
.detail .num {font-family: var(--serif); font-size: 2rem; font-weight: 600; color: var(--ink); line-height: 1; margin-right: 10px;}
.detail .sub {font-size: 0.88rem; color: var(--muted);}
.detail .verdict {font-size: 0.97rem; color: var(--ink); margin-bottom: 6px;}
a.readlink {display: inline-block; margin-top: 10px; padding: 7px 14px; border-radius: 3px; background: var(--black); color: #ffffff !important; text-decoration: none; font-weight: 500; font-size: 0.9rem;}
a.readlink:hover {background: var(--green);}

/* the big numbers on "score your own paper" */
.big {font-family: var(--serif); font-size: 2.6rem; font-weight: 600; line-height: 1.05; color: var(--ink);}

/* boxes for charts and results (st.container(key=...)) */
.st-key-why_feed, .st-key-own_result, .st-key-rising_chart {background: #ffffff; border: 1px solid var(--line); border-radius: 4px; padding: 18px 20px 12px 20px;}

/* the "how it works" sheet */
.sheet {background: #ffffff; border: 1px solid var(--line); border-top: 3px solid var(--green); border-radius: 4px; padding: 26px 32px 18px 32px; max-width: 60rem;}
.sheet p {font-size: 0.98rem; line-height: 1.55; color: var(--ink); margin: 0 0 12px 0;}
.facts {display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px 28px; border-bottom: 1px solid var(--line); padding: 0 0 14px 0; margin: 0 0 20px 0;}
.facts .k {display: block; font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--green); margin-bottom: 3px;}
.facts .v {font-size: 0.92rem; color: var(--ink); line-height: 1.45;}
div[data-testid="stMetricValue"] {font-family: var(--serif);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def scorer() -> Scorer:
    return Scorer()


@st.cache_data(show_spinner=False)
def pool_cached(mtime: float):
    return load_pool()


def pool_mtime() -> float:
    f = POOL_DIR / "biotech_pool.parquet"
    return f.stat().st_mtime if f.exists() else 0.0


@st.cache_data(show_spinner=False)
def explain_cached(paper_id: str, title: str, abstract: str, category: str, authors: str, n_authors, posted: str) -> dict:
    r = scorer().score(title, abstract, category, author_names=names_from_biorxiv(authors) or None,
                       n_authors=int(n_authors) if pd.notna(n_authors) else None, posted=pd.to_datetime(posted).date())
    return {k: r[k] for k in ["probability", "probability_content", "unknown_lab", "raw_probability", "verdict", "groups", "words_for", "words_against",
                              "neighbours", "authors", "assumptions"]}


# ------------------------------------------------------------------ small building blocks
def show(fig) -> None:
    """A chart on paper: transparent figure, so the box behind it shows through."""
    fig.patch.set_facecolor("none")
    for ax in fig.axes:
        ax.set_facecolor("none")
    fig.tight_layout(); st.pyplot(fig, width="content"); plt.close(fig)


def section(label: str, note: str = "") -> None:
    st.markdown(f'<div class="section"><span class="lbl">{label}</span><span class="note">{note}</span></div>', unsafe_allow_html=True)


def missing(p) -> bool:
    return p is None or (isinstance(p, float) and pd.isna(p))


def pill(p) -> str:
    if missing(p):
        return '<span class="pill pill-low">–</span>'
    cls = "pill-high" if p >= 0.30 else "pill-mid" if p >= 0.15 else "pill-low"
    return f'<span class="pill {cls}">{p:.0%}</span>'


def short_verdict(p) -> str:
    return "" if missing(p) else verdict(p).split(":")[0]


def lab_tag(unknown) -> str:
    if missing(unknown):
        return '<span class="tag">lab not checked</span>'
    return '<span class="tag tag-gem">unknown lab</span>' if unknown else '<span class="tag">known lab</span>'


def compact_card(r: pd.Series, score_col: str, rank: int | None = None, gem: bool = False) -> str:
    t = html.escape(r["title"] if len(r["title"]) <= 110 else r["title"][:107] + "…")
    sub = f'by content alone · with the authors {r["probability"]:.0%}' if gem else short_verdict(r[score_col])   # the verdict follows the chosen score
    no = f'<div class="rank">{rank}</div>' if rank else ""
    tag = "" if missing(r.get("unknown_lab")) else " " + lab_tag(r.get("unknown_lab"))   # "not checked" would only add noise on a small card
    return (f'<div class="card{" gem" if gem else ""}">{no}<div class="meta">{r["rxiv_category"]} · {r["posted"]}{tag}</div>'
            f'<h3>{t}</h3><div class="score">{pill(r[score_col])}<span>{sub}</span></div>'
            f'<a class="read" href="{r["url"]}" target="_blank">Read on bioRxiv →</a></div>')


def detail_card(row: pd.Series, extra: str = "") -> None:
    p_all, p_con = row.get("probability"), row.get("probability_content")
    num = lambda p: "–" if missing(p) else f"{p:.0%}"  # noqa: E731
    st.markdown(f"""
<div class="detail">
  <div class="meta">{row.get('rxiv_category', '')} · posted {row.get('posted', '')} {lab_tag(row.get('unknown_lab'))}</div>
  <h2>{html.escape(str(row['title']))}</h2>
  <div class="scores">
    <div><span class="lbl">Chance of a top 10 % paper, with the authors</span><span class="num">{num(p_all)}</span><span class="sub">{short_verdict(p_all)}</span></div>
    <div><span class="lbl">Judged by content only, blind to the authors</span><span class="num">{num(p_con)}</span><span class="sub">{short_verdict(p_con)}</span></div>
  </div>
  <div class="verdict">{'' if missing(p_all) else verdict(p_all).capitalize() + '.'} <span class="muted">The average preprint has a 10 % chance.</span></div>
  {extra}
  <a class="readlink" href="{row.get('url', '#')}" target="_blank">Read the paper on bioRxiv →</a>
</div>""", unsafe_allow_html=True)


def why_panel(r: dict) -> None:
    g = r["groups"].drop("starting point")
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ibcs.bars(ax, g.index, g.values, fmt="{:+.2f}", baseline=0, colors=[ibcs.DARK if v > 0 else ibcs.LIGHT for v in g.values])
    ibcs.title(ax, "What pushes this paper up or down (model with the authors)", "contribution to the model's score (log-odds); dark = up, light = down")
    show(fig)
    c1, c2 = st.columns(2)
    with c1:
        if len(r["words_for"]):
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ibcs.bars(ax, r["words_for"].index, r["words_for"].values, fmt="{:+.2f}")
            ibcs.title(ax, "Words that help", "the paper's own words with the largest positive weight")
            show(fig)
    with c2:
        if len(r["words_against"]):
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ibcs.bars(ax, r["words_against"].index, r["words_against"].values, fmt="{:+.2f}", colors=[ibcs.LIGHT] * len(r["words_against"]))
            ibcs.title(ax, "Words that hurt", "the paper's own words with the largest negative weight")
            show(fig)
    for au in r["authors"]:
        found = au["found"] or "not found"
        record = f"{au['prior_works']} works and {au['prior_cites']} citations before this year" if au.get("prior_cites") is not None else "track record unknown"
        st.caption(f"Author lookup: '{au['name']}' → {found}; {record}.")
    for note in r["assumptions"]:
        st.caption("Assumption: " + note)
    st.markdown("**The five most similar past papers, and what happened to them**")
    n = r["neighbours"].rename(columns={"cites_12m": "first-year citations", "top10": "top 10 %", "rxiv_category": "category"})
    n["first-year citations"] = n["first-year citations"].round(1)
    n["top 10 %"] = n["top 10 %"].map({1: "yes", 0: "no"})
    st.dataframe(n[["title", "posted", "category", "first-year citations", "top 10 %", "similarity"]], hide_index=True)


# ------------------------------------------------------------------ load
S = scorer()
exp = {**S.b["expected"], **EXPECTED}
pool, pool_E = pool_cached(pool_mtime())
pool_has_content = pool is not None and "probability_content" in pool.columns
feed_line = f"Feed built {pool['scored_on'].iloc[0]} · {len(pool):,} preprints from the last 30 days" if pool is not None and len(pool) else "The feed has not been built yet"

st.markdown(f"""
<div class="hero">
  <div class="kicker">bioRxiv · bioengineering, synthetic biology, genomics, bioinformatics, molecular biology</div>
  <h1>Paper Adoption Predictor</h1>
  <p>New biotech preprints, ranked by how likely they are to be among the 10 % most cited of their half-year.
  The model reads the paper and, unless you switch it off, the authors' track record.</p>
  <div class="meta"><span>{feed_line}</span>
  <span class="legend"><span class="pill pill-high">30 % +</span> first look &nbsp;&nbsp; <span class="pill pill-mid">15 – 29 %</span> worth a look &nbsp;&nbsp; <span class="pill pill-low">under 15 %</span> ordinary, like most preprints</span></div>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar: the switch and the filters
with st.sidebar:
    st.markdown("""<div class="side-brand"><svg width="20" height="20" viewBox="0 0 20 20"><polygon points="16.9,14 10,18 3.1,14 3.1,6 10,2 16.9,6" fill="none" stroke="#1f4d3a" stroke-width="1.8" stroke-linejoin="round"/><circle cx="10" cy="10" r="1.6" fill="#1f4d3a"/></svg>
    Paper Adoption Predictor <span class="small">biotech</span></div>""", unsafe_allow_html=True)
    st.markdown('<div class="side-lbl">The model</div>', unsafe_allow_html=True)
    blind = st.toggle("Take the authors out of the equation", value=False, disabled=not pool_has_content,
                      help="Judge the paper alone: title, abstract, length, code link, category. Nothing about who wrote it. Shows the hidden gems: strong papers from unknown labs.")
    st.markdown('<div class="side-lbl">Filters</div>', unsafe_allow_html=True)
    cats = st.multiselect("Categories", CATEGORIES, default=CATEGORIES)
    days = st.slider("Posted in the last … days", 1, 30, 14)
    min_p = st.slider("Minimum chance", 0.0, 0.6, 0.0, 0.05, format="%.2f")
    only_unknown = st.checkbox("Unknown labs only", value=False, disabled=not pool_has_content)
    newest = st.checkbox("Newest first instead of best first", value=False)
    st.markdown("---")
    if pool is not None and len(pool):
        st.markdown('<div class="side-lbl">Feed</div>', unsafe_allow_html=True)
        st.caption(f"Built on {pool['scored_on'].iloc[0]} from {len(pool):,} preprints of the last 30 days.")
        if st.button("Refresh the feed (a few minutes)"):
            with st.spinner("fetching and scoring the last 30 days of bioRxiv..."):
                build_pool(days=30, scorer=S)
            st.cache_data.clear(); st.rerun()
    st.markdown('<div class="side-foot">A reading aid, not a judge: it says what to read first.<br>Fontys ICT · data science · 2026</div>', unsafe_allow_html=True)

tab_feed, tab_own, tab_rising, tab_about = st.tabs(["Ranked preprints", "Score your own paper", "Rising topics", "How it works"])


# ------------------------------------------------------------------ the feed
def feed_page(blind: bool) -> None:
    """The ranked feed. blind = judge by content only (the sidebar switch)."""
    score_col = "probability_content" if (blind and pool_has_content) else "probability"
    if pool is None or len(pool) == 0:
        st.info("The feed has not been built yet. It fetches every bioRxiv preprint of the last 30 days in the five biotech categories and scores them (a few minutes).")
        if st.button("Build the feed now", type="primary"):
            with st.spinner("fetching and scoring the last 30 days of bioRxiv..."):
                build_pool(days=30, scorer=S)
            st.cache_data.clear(); st.rerun()
        return

    st.markdown('<div class="lead">' + ("Judged by the paper alone: title, abstract, length, code link, category. Nothing about who wrote it. Use this to find what deserves a look regardless of the lab."
               if blind else "Judged by the paper and by the authors' track record. Where the authors have not been looked up yet, an average track record is assumed.") + "</div>", unsafe_allow_html=True)

    # ---- search on the page itself
    with st.form("search", border=False):
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        q = c1.text_input("Search in plain words (optional): the papers closest in meaning are kept, then ranked", "",
                          placeholder="for example: spatial transcriptomics of tumours, protein language models, CRISPR delivery")
        go = c2.form_submit_button("Search", type="primary", width="stretch")
    query = q.strip() if (go or st.session_state.get("active_q")) and q.strip() else ""
    st.session_state["active_q"] = query

    view = pool[pool["rxiv_category"].isin(cats)].copy()
    view = view[pd.to_datetime(view["posted"]) >= pd.Timestamp(date.today() - timedelta(days=days))]
    view = view[view[score_col] >= min_p]
    if only_unknown and "unknown_lab" in view:
        view = view[view["unknown_lab"] == True]  # noqa: E712
    past = None
    if query:
        with st.spinner("searching..."):
            new, past = search(S, query, k=60, pool=pool, pool_E=pool_E)
        sims = new.set_index("paper_id")["similarity"]
        view = view[view["paper_id"].isin(sims.index)].copy()
        view["similarity"] = view["paper_id"].map(sims)
        st.caption(f"Showing the {len(view)} papers closest in meaning to '{query}'. Clear the box and search again to see everything.")
    view = view.sort_values("posted" if newest else score_col, ascending=False).reset_index(drop=True)

    # ---- top five
    section("Top five", "the best chances in this selection" + (", newest first" if newest else "") + (", judged by content only" if blind else ""))
    top = view.head(5)
    if len(top):
        for i, (col, (_, r)) in enumerate(zip(st.columns(len(top)), top.iterrows()), start=1):
            col.markdown(compact_card(r, score_col, rank=i), unsafe_allow_html=True)
    else:
        st.caption("Nothing matches these filters.")
    if blind and pool_has_content:
        gems = view[(view["probability_content"] >= 0.25) & (view["unknown_lab"] == True)].sort_values("probability_content", ascending=False)  # noqa: E712
        section("Hidden gems", "rated 25 % or more by content alone, from authors with little or no track record: the papers everybody else will miss first")
        if len(gems):
            for col, (_, r) in zip(st.columns(min(5, len(gems))), gems.head(5).iterrows()):
                col.markdown(compact_card(r, "probability_content", gem=True), unsafe_allow_html=True)
        else:
            st.caption("None in this selection yet. The lab check needs the author lookup, which runs when the feed is refreshed.")
    if past is not None and len(past):
        with st.expander("Similar papers from the past, and what happened to them"):
            shown = past.head(12).rename(columns={"cites_12m": "first-year citations", "top10": "top 10 %", "rxiv_category": "category"})
            shown["first-year citations"] = shown["first-year citations"].round(1)
            shown["top 10 %"] = shown["top 10 %"].map({1: "yes", 0: "no"})
            shown["read"] = "https://doi.org/" + shown["paper_id"].astype(str)
            st.dataframe(shown[["title", "category", "posted", "first-year citations", "top 10 %", "similarity", "read"]], hide_index=True,
                         column_config={"read": st.column_config.LinkColumn(display_text="open")})

    # ---- the full table, then the detail of the selected row
    section(f"All {len(view)} matching preprints", "click a row for the verdict, the abstract, the reasons and the link")
    cols = ["title", "rxiv_category", "posted", "probability"] + (["probability_content"] if pool_has_content else []) + (["similarity"] if query else [])
    shown = view[cols].rename(columns={"rxiv_category": "category", "probability": "with the authors", "probability_content": "content only"})
    for c in ["with the authors", "content only"]:                              # whole percents in the bars
        if c in shown:
            shown[c] = (shown[c] * 100).round(0)
    config = {"with the authors": st.column_config.ProgressColumn(format="%d %%", min_value=0, max_value=60),
              "content only": st.column_config.ProgressColumn(format="%d %%", min_value=0, max_value=60),
              "title": st.column_config.TextColumn(width="large"), "posted": st.column_config.TextColumn(width="small"), "category": st.column_config.TextColumn(width="small")}
    event = st.dataframe(shown, hide_index=True, height=420, on_select="rerun", selection_mode="single-row", column_config=config, key="table")
    rows = event.selection.rows if event and event.selection else []
    if rows:
        r = view.iloc[rows[0]]
        extra = f'<div class="muted">authors checked: {html.escape(str(r["authors_checked"]))}</div>' if r.get("authors_checked") else ""
        detail_card(r, extra)
        with st.expander("Abstract"):
            st.write(r["abstract"])
        if st.button("Why this score?", type="primary"):
            with st.spinner("taking the score apart..."):
                ex = explain_cached(r["paper_id"], r["title"], r["abstract"], r["rxiv_category"], r.get("authors", ""), r.get("n_authors"), str(r["posted"]))
            with st.container(key="why_feed"):
                why_panel(ex)


with tab_feed:
    feed_page(blind)

# ------------------------------------------------------------------ score your own
with tab_own:
    left, right = st.columns([1, 1], gap="large")
    with left:
        title = st.text_input("Title", "A foundation model for single-cell multi-omics enables atlas-scale integration across human tissues")
        abstract = st.text_area("Abstract", "We present a transformer-based foundation model trained on 40 million single cells from human atlases. "
                                "The model integrates RNA, ATAC and protein modalities, generalizes to unseen tissues and enables zero-shot cell type "
                                "annotation. Code and weights are available at https://github.com/example/model.", height=200)
        category = st.selectbox("bioRxiv category", CATEGORIES, index=CATEGORIES.index("bioinformatics"))
        authors = st.text_input("Authors, comma separated (the first and the last are looked up in OpenAlex; leave empty for an average track record)", "")
        go = st.button("Score this paper", type="primary")
    with right:
        if go and title.strip() and abstract.strip():
            with st.spinner("reading the paper, looking up the authors, comparing with 30,000 past papers..."):
                r = S.score(title, abstract, category, author_names=[a for a in authors.split(",") if a.strip()] or None)
            with st.container(key="own_result"):
                a, b = st.columns(2)
                a.markdown(f'<div class="big">{r["probability"]:.0%}</div><div class="muted">chance of a top 10 % paper, with the authors</div>', unsafe_allow_html=True)
                if r.get("probability_content") is not None:
                    b.markdown(f'<div class="big">{r["probability_content"]:.0%}</div><div class="muted">judged by content only, blind to the authors</div>', unsafe_allow_html=True)
                st.markdown(f"**Verdict:** {r['verdict']}. <span class='muted'>The average preprint has a 10 % chance; raw score before calibration {r['raw_probability']:.0%}.</span>", unsafe_allow_html=True)
                why_panel(r)
        else:
            st.markdown('<div class="lead">Paste a title and an abstract, pick the category, optionally the authors, and press the button. '
                        "You get two numbers: the chance with the authors and the chance judged by content alone, with the reasons and the five most similar past papers.</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ rising topics
with tab_rising:
    rising = S.b["rising"].copy()
    st.markdown('<div class="lead">Two-word terms whose share of biotech preprints grew most from 2023 to the first half of 2025, and how often the recent papers that use them became top papers.</div>', unsafe_allow_html=True)
    top = rising.head(20)
    with st.container(key="rising_chart"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ibcs.bars(ax, top["term"], (top["gain"] * 100).values, fmt="{:+.1f} pp", highlight=top["term"].iloc[0])
        ibcs.title(ax, f"'{rising['term'].iloc[0]}' is the fastest-rising topic", "change in the share of preprints mentioning the term, percentage points, 2023 → 2025 H1")
        show(fig)
    show_t = rising.copy()
    for c in ["share 2023", "share 2025 H1", "gain", "top 10 % share, recent"]:
        show_t[c] = (show_t[c] * 100).round(1)
    st.dataframe(show_t.rename(columns={"share 2023": "share 2023 (%)", "share 2025 H1": "share 2025 H1 (%)", "gain": "gain (pp)", "top 10 % share, recent": "top 10 % among them (%)"}), hide_index=True)
    st.caption("Rising is not the same as cited: the last column says how often the recent papers mentioning the term were top papers (10 % is the average). Rising and above 10 % = a wave; rising and below = a crowd.")

# ------------------------------------------------------------------ how it works
with tab_about:
    st.markdown(f"""
<div class="sheet">
<div class="facts">
  <div><span class="k">Data</span><span class="v">bioRxiv, five biotech categories (bioinformatics, molecular biology, genomics, bioengineering, synthetic biology), 2022-2024 · 25,412 papers · calibrated on the 5,022 papers of January-June 2025 that have citation data</span></div>
  <div><span class="k">Models</span><span class="v">With the authors: SPECTER2 embeddings + 14 day-one facts + TF-IDF words, logistic regression, Platt-scaled. Content only: the same without the author count and the authors' track record</span></div>
  <div><span class="k">Sealed exam, Jan-Jun 2025</span><span class="v">AUROC {exp['auroc']:.3f} [{exp['low']:.3f}, {exp['high']:.3f}] · the top-10 % list holds {exp['precision_top10']:.0%} real top papers (random 10 %) · {exp['winners_found']} of the 50 most cited papers were on it (random {exp['winners_random']})</span></div>
  <div><span class="k">Source</span><span class="v">Notebooks 06-14 and <code>src/paper_adoption/predict.py</code> in the project repository · charts follow IBCS</span></div>
</div>
<p><b>What is measured.</b> Adoption = citations in the first 12 months after a preprint appears, counted by OpenAlex. A paper is a <i>top paper</i> if it is
among the 10 % most cited of the papers posted in the same half-year. So the average preprint has a 10 % chance by definition; 12 % is ordinary, 30 % is three times the usual.</p>
<p><b>What the tool reads.</b> Only what exists on the posting day: title and abstract (a pretrained science model, SPECTER2, plus the words themselves),
the lengths, a code link, the bioRxiv category, and, unless you switch the authors off, the number of authors and the authors' track record before that year.
The feed looks the authors up only for the most promising papers; where they have not been looked up, an average track record is assumed (a lab with zero history would be the worst case, not the honest guess).</p>
<p><b>The switch.</b> With the authors, the score predicts what the field will actually read; it rewards known labs because the field does. With the authors taken out,
the paper is judged alone, so a strong paper from an unknown lab is not pushed down, and the hidden-gems list appears. Notebook 13 measures,
on the 2025 exam, how many top papers from unknown labs each version caught.</p>
<p><b>How it was built.</b> Seven iterations: nearest neighbours, then linear and logistic regression, then random forest and gradient boosting (same scores: the algorithm was not the limit),
then the words of the abstract (the first real jump), then SPECTER2 (a small extra), then error bars, calibration and a sealed exam, then the author-blind model.
Notebook 14 checks that the result is not just the pretrained model and that logistic regression beats four other algorithms on the same inputs.</p>
<p><b>How it was tested.</b> Always on the future: learn from 2022-2023, predict 2024. Then once on a sealed set, the bioRxiv papers of January-June 2025, that no decision was based on.</p>
<p><b>What it does not do.</b> It does not predict citation counts: no model explained more than a seventh of their variation, so the tool shows a ranking and a probability, never a number of citations.
It is a reading aid: use it to decide what to read first.</p>
<p class="muted">Built by Jean Pierre van Driel with Claude (Anthropic, 2026) as a Fontys data-science project.</p>
</div>
""", unsafe_allow_html=True)
