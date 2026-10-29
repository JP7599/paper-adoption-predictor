"""The tool: score a brand-new biotech preprint on the day it appears, and say why.

    .venv/bin/python -m paper_adoption.predict train   # fit the final model on 2022-2024, calibrate on Jan-Jun 2025, save models/biotech_adoption.joblib
    .venv/bin/python -m paper_adoption.predict demo    # score one example paper and print the explanation
    .venv/bin/python -m paper_adoption.predict digest  # this week's picks: score every bioRxiv preprint of the last 7 days, keep the best
    .venv/bin/python -m paper_adoption.predict pool 30 # the feed for the web app: every preprint of the last 30 days, scored, ready to filter and search
    .venv/bin/python -m paper_adoption.predict search spatial transcriptomics of tumours   # plain-words search over new and past papers
    .venv/bin/streamlit run app/biotech_app.py         # the web demo (local)

What the tool returns for one paper (title, abstract, bioRxiv category, optional author names):
  * the calibrated probability that the paper ends in the top 10 % of its half-year by first-year citations
  * the expected first-year citations (a rough number from the count model)
  * why: the words that push the probability up or down, the facts that push it, the topic as a whole, and the five
    most similar past papers with what happened to them
  * rising topics: the terms whose share of biotech preprints grew most from 2023 to the first half of 2025

The model is the one of iteration four (SPECTER2 embeddings + 14 day-one facts + TF-IDF words, logistic regression),
tested in iteration five on a sealed exam: AUROC 0.725 [0.701, 0.751], precision at top 10 % 0.315, Brier 0.084.
"""
from __future__ import annotations

import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import logit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from . import cache
from . import config as C
from .authors import AUTHOR_API, AUTHOR_FIELDS, prior_counts
from .prepare import CODE_LINK
from .textclean import clean_abstract

MODEL_PATH = C.ROOT / "models" / "biotech_adoption.joblib"
MEMORY_PATH = C.ROOT / "models" / "biotech_memory_embeddings.npy"
PRESTIGE = ["first_author_prior_works", "first_author_prior_cites", "last_author_prior_works", "last_author_prior_cites", "max_prior_cites"]
FACTS = ["n_authors", "title_words", "abstract_words", "has_code_link"]
NUMERIC = FACTS + ["log_" + c for c in PRESTIGE]
CATEGORIES = sorted(C.BIOTECH_CATEGORIES)
FEATURES = NUMERIC + ["cat_" + c for c in CATEGORIES]
CONTENT = [c for c in FEATURES if "prior" not in c and c != "n_authors"]   # the content-only model never sees who or how many wrote the paper
N_EMB = 768
COLS = ["paper_id", "field", "title", "abstract", "posted", "year", "half_year", "rxiv_category", "n_authors", "title_words", "abstract_words",
        "has_code_link", "first_author_prior_works", "first_author_prior_cites", "last_author_prior_works", "last_author_prior_cites",
        "max_prior_cites", "cites_12m", "top10_oa_cohort"]
EXPECTED = {"auroc": 0.725, "low": 0.700, "high": 0.750, "precision_top10": 0.313, "brier": 0.084, "base_rate": 0.10,   # the sealed-exam results of iteration five (notebook 11)
            "winners_found": 32, "winners_random": 5}                                                                      # of the 50 most cited papers of early 2025, how many were on the tool's top 10 % list


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """The 14 day-one features, exactly as in the notebooks."""
    f = frame.copy()
    f["has_code_link"] = f["has_code_link"].fillna(0).astype(int)
    for c in PRESTIGE:
        f["log_" + c] = np.log1p(f[c].fillna(0))
    cats = pd.get_dummies(f["rxiv_category"], prefix="cat", dtype=int)
    return pd.concat([f[NUMERIC].fillna(0), cats], axis=1).reindex(columns=FEATURES, fill_value=0)


def _table(name: str) -> pd.DataFrame:
    t = pd.read_csv(C.PROCESSED_DIR / name, usecols=COLS, low_memory=False)
    t = t[(t["field"] == "biotech")].dropna(subset=["cites_12m"]).copy()
    t["text"] = t["title"].fillna("") + ". " + t["abstract"].fillna("")
    t["log_cites"] = np.log1p(t["cites_12m"])
    t["top10"] = t["top10_oa_cohort"].astype(int)
    return t.reset_index(drop=True)


def _embeddings(name: str) -> tuple[np.ndarray, pd.Series]:
    E = np.load(C.PROCESSED_DIR / f"{name}_embeddings_specter2.npy")
    ids = pd.read_csv(C.PROCESSED_DIR / f"{name}_embeddings_specter2_ids.csv")["paper_id"]
    row_of = pd.Series(range(len(ids)), index=ids)
    return E, row_of[~row_of.index.duplicated()]


def _logit(p) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)


FILLER = set("""findings framework remains remain key including critical enabling enable enables potential offering offer offers particularly robust
insights insight diverse significant address addressing revealed reveal reveals highlight highlights highlighting integrating integrate integrated novel
approach approaches results result study studies suggest suggests provide provides providing shed light paves pave way important role roles understanding
underlying mechanism mechanisms new method methods data analysis based using used use demonstrate demonstrates demonstrated show shows showed identify
identified identifies associated compared comparison high low large small different various multiple specific general broad wide widely applicable
accessible scalable efficient effective powerful comprehensive systematic work paper present presents presented propose proposed proposes develop
developed developing development tool tools resource resources platform pipeline enhance enhanced enhancing improve improved improving improvement
increase increased increasing reduce reduced reducing limited limitations challenge challenges challenging crucial essential fundamental pivotal valuable
promising rapid rapidly advances advance advanced recent recently emerging future current existing available open poorly understood unclear largely
remains unknown little known""".split())


def rising_topics(texts: pd.Series, half_years: pd.Series, top10: pd.Series, top: int = 30) -> pd.DataFrame:
    """Two-word terms whose share of biotech preprints grew most from 2023 to 2025 H1, with the top-10 % rate of the recent papers that use them.
    Two-word terms, because single words that rise are mostly writing style ("findings", "framework"); filler words are excluded."""
    cv = CountVectorizer(binary=True, min_df=25, ngram_range=(2, 2), stop_words="english", max_features=80000)
    B = cv.fit_transform(texts)                                                   # 1 = the paper mentions the term
    terms = cv.get_feature_names_out()
    hy = half_years.values
    share = {h: np.asarray(B[hy == h].mean(axis=0)).ravel() for h in sorted(set(hy))}
    recent = share["2025 H1"]
    before = (share["2023 H1"] + share["2023 H2"]) / 2
    recent_mask = np.isin(hy, ["2024 H2", "2025 H1"])
    hit_rate = np.asarray(B[recent_mask].T @ top10.values[recent_mask]).ravel() / np.maximum(np.asarray(B[recent_mask].sum(axis=0)).ravel(), 1)
    out = pd.DataFrame({"term": terms, "share 2023": before, "share 2025 H1": recent, "gain": recent - before, "top 10 % share, recent": hit_rate})
    topical = out["term"].map(lambda t: not any(w in FILLER for w in t.split()))
    out = out[topical & (out["share 2025 H1"] >= 0.008)]                           # a real topic: at least 0.8 % of the recent papers
    return out.sort_values("gain", ascending=False).head(top).reset_index(drop=True)


def train_and_save() -> None:
    full, exam = _table("full_papers.csv"), _table("exam_biotech.csv")
    E_full, row_full = _embeddings("full_biotech")
    E_exam, row_exam = _embeddings("exam_biotech")
    Ef, Ee = E_full[row_full.loc[full["paper_id"]].values], E_exam[row_exam.loc[exam["paper_id"]].values]
    Xf, Xe = features(full), features(exam)

    vec = TfidfVectorizer(min_df=5, max_features=30000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(full["text"])
    scaler = StandardScaler().fit(Xf)

    def A(frame, X, Emb):
        return hstack([csr_matrix(np.hstack([Emb, scaler.transform(X)])), vec.transform(frame["text"])]).tocsr()

    Af, Ae = A(full, Xf, Ef), A(exam, Xe, Ee)
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(Af, full["top10"])    # the overall ranker, learned on 2022-2024
    reg = Ridge(alpha=3.0).fit(Af, full["log_cites"])                                                 # the count model (kept for the notebooks; not shown in the app)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(clf.predict_proba(Ae)[:, 1]), exam["top10"])   # honest probabilities, fitted on 2025 H1

    scaler_c = StandardScaler().fit(Xf[CONTENT])                                                      # the content-only model: blind to the authors

    def A_content(frame, X, Emb):
        return hstack([csr_matrix(np.hstack([Emb, scaler_c.transform(X[CONTENT])])), vec.transform(frame["text"])]).tocsr()

    Acf, Ace = A_content(full, Xf, Ef), A_content(exam, Xe, Ee)
    clf_c = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(Acf, full["top10"])
    platt_c = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(clf_c.predict_proba(Ace)[:, 1]), exam["top10"])
    cutoff = float(np.nanpercentile(full["max_prior_cites"].fillna(0), 25))                            # "unknown lab": best author below the lowest quarter of training papers
    fill = {c: float(np.nanmedian(full[c].fillna(0))) for c in PRESTIGE}                                # the typical training paper, used when the authors are not looked up
    print(f"trained on {len(full)} papers (2022-2024), calibrated on {len(exam)} papers (2025 H1); unknown-lab cutoff = {cutoff:.0f} prior citations", flush=True)

    both = pd.concat([full, exam], ignore_index=True)
    memory = both[["paper_id", "title", "posted", "rxiv_category", "cites_12m", "top10"]].copy()
    np.save(MEMORY_PATH, np.vstack([Ef, Ee]).astype(np.float32))
    rising = rising_topics(both["text"], both["half_year"], both["top10"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"vec": vec, "scaler": scaler, "clf": clf, "reg": reg, "platt": platt, "features": FEATURES, "memory": memory, "rising": rising,
                 "content": {"scaler": scaler_c, "clf": clf_c, "platt": platt_c, "features": CONTENT}, "unknown_lab_cutoff": cutoff, "author_fill": fill,
                 "trained_on": f"bioRxiv 2022-2024, {len(full):,} papers; calibrated on January-June 2025, {len(exam):,} papers",
                 "expected": EXPECTED, "saved": str(date.today())}, MODEL_PATH)
    print("saved", MODEL_PATH, "and", MEMORY_PATH, flush=True)


class Scorer:
    """Load the saved tool once, then score papers."""

    def __init__(self) -> None:
        self.b = joblib.load(MODEL_PATH)
        self.memory_E = np.load(MEMORY_PATH)
        self._encoder = None

    _lock = __import__("threading").Lock()

    @property
    def encoder(self):
        with Scorer._lock:                                  # two web sessions must not load the model at the same time
            if self._encoder is None:
                from .embed import load_model
                self._encoder = load_model("specter2")    # (model, joiner); loaded on first use, about 10 seconds
        return self._encoder

    def embed(self, title: str, abstract: str) -> np.ndarray:
        model, joiner = self.encoder
        return model.encode([title + joiner + abstract], normalize_embeddings=True)[0]

    openalex_ok = True          # set to False for the rest of the session when OpenAlex rate-limits us (HTTP 429)

    def author(self, name: str, posted: date) -> dict:
        """Look an author up in OpenAlex by name and count works and citations before the posting year. Fails softly: unavailable = None, and the scorer then assumes an average track record."""
        if not Scorer.openalex_ok:
            return {"name": name, "found": None, "prior_works": None, "prior_cites": None, "note": "OpenAlex unavailable (rate limit), track record unknown"}
        try:
            j = cache.get_json(AUTHOR_API, {"search": name, "select": AUTHOR_FIELDS, "per_page": 1}, pace_seconds=0.5, retries=3, not_found_ok=True)
        except Exception as e:                                   # the daily budget or a rate limit: continue without the track record
            Scorer.openalex_ok = False
            print(f"OpenAlex author lookup unavailable ({str(e)[:60]}...), continuing without the track record", flush=True)
            return {"name": name, "found": None, "prior_works": None, "prior_cites": None, "note": "OpenAlex unavailable (rate limit), track record unknown"}
        hits = (j or {}).get("results") or []
        if not hits:
            return {"name": name, "found": None, "prior_works": 0, "prior_cites": 0}
        works, cites = prior_counts(hits[0], posted)
        return {"name": name, "found": hits[0].get("display_name"), "prior_works": works or 0, "prior_cites": cites or 0}

    def _track(self, authors: list[dict]) -> tuple[dict, bool]:
        """The five track-record inputs. Unknown (no names given, OpenAlex down) = the typical training paper, not zero: zero would mean
        'a lab that never published anything', the worst case, and it pushed every unchecked paper down. Returns (inputs, all authors checked)."""
        fill = self.b.get("author_fill") or {c: 0 for c in PRESTIGE}
        first = authors[0] if authors else {}
        last = authors[-1] if authors else {}

        def pick(a, key, col):
            v = a.get(key)
            return fill[col] if v is None else v

        t = {"first_author_prior_works": pick(first, "prior_works", "first_author_prior_works"), "first_author_prior_cites": pick(first, "prior_cites", "first_author_prior_cites"),
             "last_author_prior_works": pick(last, "prior_works", "last_author_prior_works"), "last_author_prior_cites": pick(last, "prior_cites", "last_author_prior_cites")}
        known = [a["prior_cites"] for a in authors if a.get("prior_cites") is not None]
        checked = bool(authors) and len(known) == len(authors)
        t["max_prior_cites"] = max(known) if checked else max(known + [fill["max_prior_cites"]])
        return t, checked

    def score(self, title: str, abstract: str, category: str, author_names: list[str] | None = None, n_authors: int | None = None,
              has_code_link: bool | None = None, posted: date | None = None) -> dict:
        posted = posted or date.today()
        abstract = clean_abstract(abstract)                     # the same cleaning as the training data
        names = [n.strip() for n in (author_names or []) if n.strip()]
        assumptions = []
        if n_authors is None:
            n_authors = len(names) if names else 6
            if not names:
                assumptions.append("number of authors unknown, assumed 6 (the typical biotech preprint)")
        if has_code_link is None:
            has_code_link = bool(CODE_LINK.search(abstract))
        authors = [self.author(n, posted) for n in ([names[0]] + ([names[-1]] if len(names) > 1 else []))] if names else []
        if not names:
            assumptions.append("no author names given, so an average track record is assumed")
        if any(a.get("note") for a in authors):
            assumptions.append("OpenAlex was unavailable (rate limit), so an average track record is assumed")
        track, checked = self._track(authors)
        row = pd.DataFrame([{"n_authors": n_authors, "title_words": len(title.split()), "abstract_words": len(abstract.split()), "has_code_link": int(has_code_link),
                             **track, "rxiv_category": category}])
        X = features(row)
        emb = self.embed(title, abstract)
        text = title + ". " + abstract
        W = self.b["vec"].transform([text])
        scaled = self.b["scaler"].transform(X)
        A = hstack([csr_matrix(np.hstack([emb[None, :], scaled])), W]).tocsr()

        clf, reg, platt = self.b["clf"], self.b["reg"], self.b["platt"]
        raw = float(clf.predict_proba(A)[0, 1])
        prob = float(platt.predict_proba(_logit([raw]))[0, 1])
        cites = float(np.expm1(reg.predict(A)[0]))
        prob_content, unknown_lab = None, None
        if "content" in self.b:                                                   # the author-blind score
            c = self.b["content"]
            Ac = hstack([csr_matrix(np.hstack([emb[None, :], c["scaler"].transform(X[c["features"]])])), W]).tocsr()
            prob_content = float(c["platt"].predict_proba(_logit([c["clf"].predict_proba(Ac)[0, 1]]))[0, 1])
            if checked:
                unknown_lab = bool(track["max_prior_cites"] < self.b.get("unknown_lab_cutoff", 0))

        coef = clf.coef_[0]
        n_f = len(FEATURES)
        topic = float(coef[:N_EMB] @ emb)
        fact_contrib = pd.Series(coef[N_EMB:N_EMB + n_f] * scaled[0], index=FEATURES)
        w = W.toarray()[0]
        present = np.nonzero(w)[0]
        word_contrib = pd.Series(coef[N_EMB + n_f:][present] * w[present], index=self.b["vec"].get_feature_names_out()[present]).sort_values()
        groups = pd.Series({
            "topic (what the paper is about)": topic,
            "words of the abstract": float(word_contrib.sum()),
            "authors' track record": float(fact_contrib[[c for c in FEATURES if "prior" in c]].sum()),
            "paper facts (length, authors, code, category)": float(fact_contrib[[c for c in FEATURES if "prior" not in c]].sum()),
            "starting point": float(clf.intercept_[0]),
        })
        sims = self.memory_E @ emb
        idx = np.argsort(-sims)[:5]
        neighbours = self.b["memory"].iloc[idx].assign(similarity=sims[idx].round(3)).reset_index(drop=True)
        return {"probability": prob, "probability_content": prob_content, "unknown_lab": unknown_lab, "raw_probability": raw, "expected_citations": cites,
                "verdict": verdict(prob), "groups": groups,
                "words_for": word_contrib[word_contrib > 0].sort_values(ascending=False).head(8), "words_against": word_contrib[word_contrib < 0].head(8),
                "facts": fact_contrib.sort_values(ascending=False), "neighbours": neighbours, "authors": authors, "assumptions": assumptions,
                "inputs": {"n_authors": n_authors, "has_code_link": has_code_link, "category": category, "posted": str(posted)}}


    def score_many(self, frame: pd.DataFrame, return_embeddings: bool = False):
        """Score a table of papers (title, abstract, rxiv_category, n_authors) in one go, without the author lookup (an average track record is assumed). Fast: one embedding batch."""
        f = frame.copy()
        f["abstract"] = f["abstract"].fillna("").map(clean_abstract)
        model, joiner = self.encoder
        emb = model.encode((f["title"].fillna("") + joiner + f["abstract"]).tolist(), batch_size=32, normalize_embeddings=True)
        rows = pd.DataFrame({"n_authors": f["n_authors"].fillna(6).astype(int).values, "title_words": f["title"].fillna("").str.split().str.len().values,
                             "abstract_words": f["abstract"].str.split().str.len().values, "has_code_link": f["abstract"].str.contains(CODE_LINK).astype(int).values,
                             **{c: (self.b.get("author_fill") or {}).get(c, 0) for c in PRESTIGE}, "rxiv_category": f["rxiv_category"].values})
        X = features(rows)
        A = hstack([csr_matrix(np.hstack([emb, self.b["scaler"].transform(X)])), self.b["vec"].transform((f["title"].fillna("") + ". " + f["abstract"]).tolist())]).tocsr()
        raw = self.b["clf"].predict_proba(A)[:, 1]
        f["probability"] = self.b["platt"].predict_proba(_logit(raw))[:, 1]
        f["expected_citations"] = np.expm1(self.b["reg"].predict(A))
        if "content" in self.b:
            c = self.b["content"]
            Ac = hstack([csr_matrix(np.hstack([emb, c["scaler"].transform(X[c["features"]])])), self.b["vec"].transform((f["title"].fillna("") + ". " + f["abstract"]).tolist())]).tocsr()
            f["probability_content"] = c["platt"].predict_proba(_logit(c["clf"].predict_proba(Ac)[:, 1]))[:, 1]
        return (f, emb) if return_embeddings else f


def digest(days: int = 7, categories: list[str] | None = None, top: int = 15, refine: int = 40, scorer: "Scorer | None" = None, save: bool = True) -> pd.DataFrame:
    """This week's picks: every bioRxiv preprint of the last `days` days in the biotech categories, scored; the `top` best are returned.
    Two passes: all papers without the author lookup (fast), then the `refine` best again with the first and last author looked up in OpenAlex."""
    from datetime import timedelta
    from . import biorxiv
    s = scorer or Scorer()
    today = date.today()
    start = today - timedelta(days=days)
    rows = []
    for cat in (categories or CATEGORIES):
        rows += biorxiv.first_versions(biorxiv.list_interval("biorxiv", start, today, category=cat), "biorxiv", "biotech")
    papers = pd.DataFrame(rows).drop_duplicates("paper_id")
    papers = papers[papers["abstract"].fillna("").str.split().str.len() > 50].reset_index(drop=True)
    if papers.empty:
        return papers
    scored = s.score_many(papers)
    best = scored.nlargest(min(refine, len(scored)), "probability").copy()
    refined, found = [], []
    for _, p in best.iterrows():                                                 # second pass: with the authors' track record
        r = s.score(p["title"], p["abstract"], p["rxiv_category"], author_names=names_from_biorxiv(p.get("authors", "")) or None,
                    n_authors=int(p["n_authors"]) if pd.notna(p.get("n_authors")) else None, posted=pd.to_datetime(p["posted"]).date())
        refined.append(r["probability"]); found.append("; ".join(f"{a['found'] or a['name'] + ' (not found)'}" for a in r["authors"]))
    best["probability"] = refined
    best["authors_checked"] = found
    out = best.sort_values("probability", ascending=False).head(top)
    out["verdict"] = out["probability"].map(verdict)
    out["url"] = "https://www.biorxiv.org/content/" + out["paper_id"].astype(str) + "v1"
    out = out[["title", "rxiv_category", "posted", "probability", "expected_citations", "verdict", "authors_checked", "url", "paper_id"]].reset_index(drop=True)
    out.attrs["scanned"] = len(papers); out.attrs["from"] = str(start); out.attrs["to"] = str(today)
    if save:
        (C.ROOT / "reports").mkdir(exist_ok=True)
        out.assign(scanned=len(papers), window_from=str(start), window_to=str(today)).to_csv(C.ROOT / "reports" / f"biotech_digest_{today}.csv", index=False)
    return out


POOL_DIR = C.ROOT / "reports" / "pool"


def build_pool(days: int = 30, refine: int = 60, scorer: "Scorer | None" = None) -> pd.DataFrame:
    """The feed: every bioRxiv preprint of the last `days` days in the five biotech categories, scored, embeddings kept for the search.
    The `refine` most promising papers also get the author lookup. Saves reports/pool/biotech_pool.parquet and biotech_pool_embeddings.npy."""
    from datetime import timedelta
    from . import biorxiv
    s = scorer or Scorer()
    today = date.today()
    start = today - timedelta(days=days)
    rows = []
    for cat in CATEGORIES:
        rows += biorxiv.first_versions(biorxiv.list_interval("biorxiv", start, today, category=cat), "biorxiv", "biotech")
    papers = pd.DataFrame(rows).drop_duplicates("paper_id")
    papers = papers[papers["abstract"].fillna("").str.split().str.len() > 50].reset_index(drop=True)
    if papers.empty:
        return papers
    scored, emb = s.score_many(papers, return_embeddings=True)
    scored["authors_checked"] = ""
    scored["unknown_lab"] = None
    promising = set(scored["probability"].nlargest(min(refine, len(scored))).index)
    if "probability_content" in scored:
        promising |= set(scored["probability_content"].nlargest(min(refine, len(scored))).index)   # also the best by content, to find hidden gems
    for i in sorted(promising):                                                        # second pass for the most promising: who wrote it?
        p = scored.loc[i]
        r = s.score(p["title"], p["abstract"], p["rxiv_category"], author_names=names_from_biorxiv(p.get("authors", "")) or None,
                    n_authors=int(p["n_authors"]) if pd.notna(p.get("n_authors")) else None, posted=pd.to_datetime(p["posted"]).date())
        scored.loc[i, "probability"], scored.loc[i, "expected_citations"] = r["probability"], r["expected_citations"]
        scored.loc[i, "authors_checked"] = "; ".join((a["found"] or a["name"] + " (not found)") for a in r["authors"])
        scored.loc[i, "unknown_lab"] = r["unknown_lab"]
    scored["verdict"] = scored["probability"].map(verdict)
    scored["url"] = "https://www.biorxiv.org/content/" + scored["paper_id"].astype(str) + "v1"
    scored["scored_on"] = str(today)
    order = scored["probability"].sort_values(ascending=False).index
    keep = ["paper_id", "title", "abstract", "authors", "n_authors", "posted", "rxiv_category", "probability", "probability_content", "unknown_lab",
            "expected_citations", "verdict", "authors_checked", "url", "scored_on"]
    keep = [k for k in keep if k in scored]
    pool = scored.loc[order, keep].reset_index(drop=True)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(POOL_DIR / "biotech_pool.parquet", index=False)
    np.save(POOL_DIR / "biotech_pool_embeddings.npy", emb[order.values].astype(np.float32))
    print(f"feed: {len(pool)} preprints posted {start} to {today}, saved in {POOL_DIR}", flush=True)
    return pool


def rescore_pool(scorer: "Scorer | None" = None) -> pd.DataFrame:
    """Rescore the saved feed with the current scorer, without refetching bioRxiv and without author lookups
    (the lab check is reset; the next `pool` run redoes it). Used after a change to the scorer."""
    s = scorer or Scorer()
    pool = pd.read_parquet(POOL_DIR / "biotech_pool.parquet")
    scored, emb = s.score_many(pool, return_embeddings=True)
    for c in ["probability", "probability_content", "expected_citations"]:
        if c in scored:
            pool[c] = scored[c].values
    pool["verdict"] = pool["probability"].map(verdict)
    pool["authors_checked"], pool["unknown_lab"] = "", None
    pool.to_parquet(POOL_DIR / "biotech_pool.parquet", index=False)
    np.save(POOL_DIR / "biotech_pool_embeddings.npy", emb.astype(np.float32))
    print(f"feed rescored: {len(pool)} preprints, mean chance {pool['probability'].mean():.1%} overall, {pool['probability_content'].mean():.1%} by content", flush=True)
    return pool


def load_pool():
    """(pool table, embeddings) or (None, None) if the feed was never built."""
    f = POOL_DIR / "biotech_pool.parquet"
    if not f.exists():
        return None, None
    return pd.read_parquet(f), np.load(POOL_DIR / "biotech_pool_embeddings.npy")


def search(scorer: "Scorer", query: str, k: int = 20, pool: pd.DataFrame | None = None, pool_E: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plain-words search. The query is embedded with SPECTER2 and compared with every paper: returns (new papers from the feed, with their
    probability) and (past papers from the memory, with what happened to them), both with a similarity from 0 to 1."""
    q = scorer.embed(query, "")
    new = pd.DataFrame()
    if pool is not None and len(pool):
        sims = pool_E @ q
        top = np.argsort(-sims)[:k]
        new = pool.iloc[top].assign(similarity=sims[top].round(3)).reset_index(drop=True)
    sims = scorer.memory_E @ q
    top = np.argsort(-sims)[:k]
    past = scorer.b["memory"].iloc[top].assign(similarity=sims[top].round(3)).reset_index(drop=True)
    return new, past


def latest_digest() -> pd.DataFrame | None:
    files = sorted((C.ROOT / "reports").glob("biotech_digest_*.csv"))
    return pd.read_csv(files[-1]) if files else None


def names_from_biorxiv(authors: str) -> list[str]:
    """bioRxiv lists authors as 'Lastname, Initials; Lastname, Initials'. Return 'Initials Lastname' names, in order."""
    out = []
    for part in str(authors or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "," in part:
            last, first = [x.strip() for x in part.split(",", 1)]
            part = f"{first} {last}".strip()
        out.append(part)
    return out


def verdict(p: float) -> str:
    if p >= 0.30:
        return "strong candidate for a first look: about three times the usual chance of becoming a top paper"
    if p >= 0.15:
        return "worth a look: above the usual 1-in-10 chance"
    return "ordinary: about the usual 1-in-10 chance, like most preprints"


def demo() -> None:
    s = Scorer()
    r = s.score("A foundation model for single-cell multi-omics enables atlas-scale integration across human tissues",
                "We present a transformer-based foundation model trained on 40 million single cells from human atlases. "
                "The model integrates RNA, ATAC and protein modalities, generalizes to unseen tissues and enables zero-shot "
                "cell type annotation. Code and weights are available at https://github.com/example/model.",
                "bioinformatics", author_names=["Aviv Regev", "Fabian Theis"])
    print(f"probability of top 10 %: {r['probability']:.2f} (raw {r['raw_probability']:.2f}) | expected first-year citations: {r['expected_citations']:.1f}")
    print("verdict:", r["verdict"]); print(r["groups"].round(2)); print("words for:", list(r["words_for"].index)); print("words against:", list(r["words_against"].index))
    print(r["neighbours"][["title", "posted", "cites_12m", "top10", "similarity"]])
    print("authors:", r["authors"]); print("assumptions:", r["assumptions"])


if __name__ == "__main__":
    def _digest_cli():
        d = digest()
        print(f"scanned {d.attrs.get('scanned', '?')} preprints posted {d.attrs.get('from')} to {d.attrs.get('to')}; the top {len(d)}:")
        print(d[["title", "rxiv_category", "probability", "verdict"]].to_string(index=False, max_colwidth=70))
    def _pool_cli():
        build_pool(days=int(sys.argv[2]) if len(sys.argv) > 2 else 30)

    def _search_cli():
        s = Scorer(); pool, E = load_pool()
        new, past = search(s, " ".join(sys.argv[2:]) or "single cell foundation model", pool=pool, pool_E=E, k=8)
        print("new papers:"); print(new[["title", "posted", "probability", "similarity"]].to_string(index=False, max_colwidth=70) if len(new) else "  (build the feed first: predict pool)")
        print("past papers:"); print(past[["title", "posted", "cites_12m", "top10", "similarity"]].to_string(index=False, max_colwidth=70))
    {"train": train_and_save, "demo": demo, "digest": _digest_cli, "pool": _pool_cli, "rescore": rescore_pool, "search": _search_cli}[(sys.argv[1:] or ["demo"])[0]]()
