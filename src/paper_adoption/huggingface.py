"""Hugging Face papers API: for an arXiv paper, the GitHub repository the community linked to it and its star count.

This replaces Papers with Code, which closed in 2025. Free, no key; the server allows 500 requests per 5 minutes,
so we pace at one request per 0.7 seconds. A 404 means "no page for this paper" and is a valid answer.

Reference: Hugging Face. (2026). Hugging Face Hub API: Papers [Application programming interface]. https://huggingface.co/docs/hub/api
"""
from __future__ import annotations

from . import cache

API = "https://huggingface.co/api/papers"


def paper(arxiv_id: str) -> dict:
    """The handful of fields we keep; all None when Hugging Face has no page for the paper."""
    j = cache.get_json(f"{API}/{arxiv_id}", pace_seconds=0.7, not_found_ok=True)
    if not j:
        return {"hf_page": 0, "hf_repo": None, "hf_stars": None, "hf_upvotes": None, "hf_models": None, "hf_datasets": None}
    return {"hf_page": 1, "hf_repo": j.get("githubRepo") or None, "hf_stars": j.get("githubStars"),
            "hf_upvotes": j.get("upvotes"), "hf_models": j.get("numTotalModels"), "hf_datasets": j.get("numTotalDatasets")}
