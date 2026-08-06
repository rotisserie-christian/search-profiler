"""
Hybrid review theme extraction for scatterplots.

LLM proposes a small set of pointed theme labels from a balanced sample.
Local embeddings assign all review sentences to those themes and score them.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .cluster_nlp import (
    MODEL_NAME,
    MAX_EXAMPLES,
    build_sentence_units,
    flatten_reviews,
    split_sentences,
)

MAX_THEMES = 35
MIN_UNIQUE_REVIEWS = 2
ASSIGN_SIMILARITY = 0.38
MAX_PROMPT_REVIEWS = 120
SNIPPET_CHARS = 280
MIN_PREVALENCE = 0.5

GENERIC_PRAISE = re.compile(
    r"\b("
    r"highly recommend|recommend(ed|ing)? this|excellent service|"
    r"great (job|service|work|experience)|amazing (job|work|service)|"
    r"fantastic|wonderful|outstanding (job|work|service)|"
    r"pleasure to work|best (company|landscaper)|five stars|5 stars"
    r")\b",
    re.I,
)

# Pointed landscaping themes used when the LLM is unavailable.
SEED_THEMES: List[Dict] = [
    {"feedback": "Missed deadlines", "type": "Weakness"},
    {"feedback": "Poor communication", "type": "Weakness"},
    {"feedback": "Unresponsive after deposit", "type": "Weakness"},
    {"feedback": "Broken promises", "type": "Weakness"},
    {"feedback": "Poor workmanship", "type": "Weakness"},
    {"feedback": "Property damage", "type": "Weakness"},
    {"feedback": "Incorrect materials installed", "type": "Weakness"},
    {"feedback": "Bad grading drainage", "type": "Weakness"},
    {"feedback": "Messy unfinished site", "type": "Weakness"},
    {"feedback": "Hidden extra charges", "type": "Weakness"},
    {"feedback": "Overpriced quote", "type": "Weakness"},
    {"feedback": "Unprofessional crew", "type": "Weakness"},
    {"feedback": "Ignored change requests", "type": "Weakness"},
    {"feedback": "Permit handling issues", "type": "Weakness"},
    {"feedback": "Abandoned incomplete job", "type": "Weakness"},
    {"feedback": "Hard to reach owner", "type": "Weakness"},
    {"feedback": "Punctual reliable crew", "type": "Strength"},
    {"feedback": "Clear project communication", "type": "Strength"},
    {"feedback": "Quality craftsmanship", "type": "Strength"},
    {"feedback": "Clean tidy job site", "type": "Strength"},
    {"feedback": "Strong design vision", "type": "Strength"},
    {"feedback": "Fair transparent pricing", "type": "Strength"},
    {"feedback": "Finished on schedule", "type": "Strength"},
    {"feedback": "Helpful material guidance", "type": "Strength"},
    {"feedback": "Respectful of property", "type": "Strength"},
    {"feedback": "Responsive project manager", "type": "Strength"},
    {"feedback": "Beautiful yard transformation", "type": "Strength"},
    {"feedback": "Attention to detail", "type": "Strength"},
    {"feedback": "Efficient hardscape install", "type": "Strength"},
    {"feedback": "Good plant selection", "type": "Strength"},
    {"feedback": "Fixed issues promptly", "type": "Strength"},
    {"feedback": "Professional quote process", "type": "Strength"},
    {"feedback": "Winter damage concerns", "type": "Weakness"},
    {"feedback": "Slow to start project", "type": "Weakness"},
    {"feedback": "Leftover debris cleanup", "type": "Mixed"},
]


def _cluster_type(avg_rating: float, hinted: Optional[str] = None) -> str:
    if hinted in ("Strength", "Weakness", "Mixed"):
        # Keep LLM hint only when it agrees with the score band; otherwise trust score
        if hinted == "Strength" and avg_rating >= 3.5:
            return "Strength"
        if hinted == "Weakness" and avg_rating <= 2.5:
            return "Weakness"
    if avg_rating >= 4.0:
        return "Strength"
    if avg_rating <= 2.0:
        return "Weakness"
    return "Mixed"


def _truncate(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return (cut or text[: limit - 1]) + "…"


def _is_generic_praise(text: str, rating: float) -> bool:
    if rating < 4:
        return False
    return bool(GENERIC_PRAISE.search(text)) and len(text.split()) < 28


def build_llm_sample(
    reviews: List[Dict],
    max_reviews: int = MAX_PROMPT_REVIEWS,
) -> List[str]:
    """
    Build a compact, rating-balanced sample for the LLM theme prompt.
    Prefer all low/mid ratings; fill remaining slots with non-generic highs.
    """
    lows = [r for r in reviews if r["rating"] <= 2]
    mids = [r for r in reviews if 2 < r["rating"] < 4]
    highs = [r for r in reviews if r["rating"] >= 4]
    highs = [r for r in highs if not _is_generic_praise(r["snippet"], r["rating"])]

    # Deterministic order: keep original order within bands
    selected: List[Dict] = []
    selected.extend(lows)
    selected.extend(mids)

    remaining = max(0, max_reviews - len(selected))
    if remaining:
        # Spread highs across the list rather than only the first page
        step = max(1, len(highs) // remaining) if highs else 1
        picked = highs[::step][:remaining]
        selected.extend(picked)

    selected = selected[:max_reviews]

    lines = []
    for r in selected:
        # Prefer a few concrete sentences over the full essay
        sentences = split_sentences(r["snippet"])
        if sentences:
            body = " ".join(sentences[:3])
        else:
            body = r["snippet"]
        lines.append(f"({r['rating']:.0f} stars) {_truncate(body)}")
    return lines


def parse_theme_response(raw: str) -> List[Dict]:
    """Parse LLM JSON theme list; tolerate markdown fences."""
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # If the model wrapped extra prose, grab the outermost JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("LLM theme response was not a JSON list")

    themes = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        label = (item.get("feedback") or item.get("label") or "").strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        hinted = (item.get("type") or "").strip().title()
        if hinted not in ("Strength", "Weakness", "Mixed"):
            hinted = None
        themes.append({"feedback": label, "type": hinted})
        if len(themes) >= MAX_THEMES:
            break
    return themes


def assign_and_score(
    themes: List[Dict],
    reviews: List[Dict],
    similarity_threshold: float = ASSIGN_SIMILARITY,
    model_name: str = MODEL_NAME,
    max_themes: int = MAX_THEMES,
    low_rating_max: float = 2.0,
) -> Tuple[Dict, List[Dict]]:
    """
    Assign sentences to themes via embeddings and build a dual-view scatter payload.

    Primary `points` = pain view (weaknesses; y = prevalence among ≤2★ reviews).
    `market.points` = full theme map (y = prevalence among all reviews).
    """
    units = build_sentence_units(reviews)
    total_reviews = len(reviews)
    low_review_ids = {r["review_id"] for r in reviews if r["rating"] <= low_rating_max}
    n_low = len(low_review_ids)

    market_axes = {
        "x": {"field": "score", "label": "Average review score", "domain": [1, 5]},
        "y": {
            "field": "prevalence",
            "label": "Prevalence among all reviews (%)",
            "domain": [0, 100],
        },
    }
    pain_axes = {
        "x": {"field": "score", "label": "Average review score", "domain": [1, 5]},
        "y": {
            "field": "prevalence",
            "label": f"Prevalence among ≤{low_rating_max:.0f}★ reviews (%)",
            "domain": [0, 100],
        },
    }

    if not themes or not units:
        payload = {
            "meta": {
                "method": "hybrid llm themes + embedding assignment",
                "primary_view": "pain",
                "total_reviews": total_reviews,
                "low_reviews": n_low,
                "total_sentences": len(units),
                "clusters_kept": 0,
            },
            "axes": pain_axes,
            "points": [],
            "market": {"axes": market_axes, "points": []},
        }
        return payload, []

    print(f"Loading {model_name} for theme assignment...")
    model = SentenceTransformer(model_name)
    theme_labels = [t["feedback"] for t in themes]
    theme_emb = np.asarray(model.encode(theme_labels, show_progress_bar=False))
    unit_texts = [u["text"] for u in units]
    unit_emb = np.asarray(model.encode(unit_texts, show_progress_bar=True))

    theme_norm = theme_emb / (np.linalg.norm(theme_emb, axis=1, keepdims=True) + 1e-9)
    unit_norm = unit_emb / (np.linalg.norm(unit_emb, axis=1, keepdims=True) + 1e-9)
    sims = unit_norm @ theme_norm.T

    best_theme = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    members: Dict[int, List[int]] = defaultdict(list)
    for i, (t_idx, sim) in enumerate(zip(best_theme, best_sim)):
        if float(sim) < similarity_threshold:
            continue
        hinted = themes[int(t_idx)].get("type")
        rating = units[i]["rating"]
        if hinted == "Weakness" and rating >= 4:
            continue
        if hinted == "Strength" and rating <= 2:
            continue
        members[int(t_idx)].append(i)

    market_points = []
    for t_idx, theme in enumerate(themes):
        idxs = members.get(t_idx, [])
        if not idxs:
            continue
        unique_reviews = {units[i]["review_id"] for i in idxs}
        if len(unique_reviews) < MIN_UNIQUE_REVIEWS:
            continue

        ratings = [units[i]["rating"] for i in idxs]
        avg_score = float(np.mean(ratings))
        prevalence_all = (
            round((len(unique_reviews) / total_reviews) * 100, 1) if total_reviews else 0.0
        )
        low_hits = unique_reviews & low_review_ids
        prevalence_low = (
            round((len(low_hits) / n_low) * 100, 1) if n_low else 0.0
        )
        if prevalence_all < MIN_PREVALENCE and prevalence_low < MIN_PREVALENCE:
            continue

        label = theme["feedback"]
        theme_type = _cluster_type(avg_score, theme.get("type"))

        ranked = sorted(idxs, key=lambda i: float(sims[i, t_idx]), reverse=True)
        examples = []
        seen = set()
        for i in ranked:
            rid = units[i]["review_id"]
            if rid in seen:
                continue
            examples.append(units[i]["text"])
            seen.add(rid)
            if len(examples) >= MAX_EXAMPLES:
                break

        # Severity: how far below neutral (3★); useful for point size on pain charts
        severity = round(max(0.0, 3.0 - avg_score), 2)

        market_points.append({
            "id": t_idx,
            "label": label,
            "x": round(avg_score, 2),
            "y": prevalence_all,
            "score": round(avg_score, 2),
            "prevalence": prevalence_all,
            "prevalence_all": prevalence_all,
            "prevalence_among_low": prevalence_low,
            "low_review_count": len(low_hits),
            "feedback": label,
            "impact_score": round(avg_score, 1),
            "type": theme_type,
            "severity": severity,
            "sentence_count": len(idxs),
            "review_count": len(unique_reviews),
            "examples": examples,
        })

    market_points.sort(key=lambda p: (-p["prevalence"], p["score"]))
    market_points = market_points[:max_themes]
    for i, point in enumerate(market_points):
        point["id"] = i

    # Pain view: actionable negativity — Weakness (and harsh Mixed), y among lows
    pain_points = []
    for p in market_points:
        if p["type"] == "Weakness" or (p["type"] == "Mixed" and p["score"] <= 2.5):
            if p["low_review_count"] < MIN_UNIQUE_REVIEWS:
                continue
            pain = dict(p)
            pain["y"] = p["prevalence_among_low"]
            pain["prevalence"] = p["prevalence_among_low"]
            pain_points.append(pain)

    pain_points.sort(key=lambda p: (-p["prevalence"], p["score"]))
    for i, point in enumerate(pain_points):
        point["id"] = i

    payload = {
        "meta": {
            "method": "hybrid llm themes + embedding assignment",
            "primary_view": "pain",
            "views": {
                "pain": (
                    "Weakness themes; y = share of ≤2★ reviews that mention the theme"
                ),
                "market": (
                    "All themes; y = share of all fetched reviews that mention the theme"
                ),
            },
            "embedding_model": model_name,
            "assign_similarity": similarity_threshold,
            "themes_proposed": len(themes),
            "total_reviews": total_reviews,
            "low_reviews": n_low,
            "low_rating_max": low_rating_max,
            "total_sentences": len(units),
            "clusters_kept": len(pain_points),
            "market_clusters": len(market_points),
            "max_themes": max_themes,
        },
        "axes": pain_axes,
        "points": pain_points,
        "market": {
            "axes": market_axes,
            "points": market_points,
        },
    }
    return payload, pain_points


def run_seeded_clustering(
    raw_data: List[Dict],
    source_path: Optional[str] = None,
    max_themes: int = MAX_THEMES,
    themes: Optional[List[Dict]] = None,
) -> Tuple[Dict, List[Dict]]:
    """Assign reviews to pointed seed themes (no LLM) and score for scatterplots."""
    reviews = flatten_reviews(raw_data)
    theme_list = themes or SEED_THEMES
    print(f"Using {len(theme_list)} seeded themes (local, no LLM)")
    payload, points = assign_and_score(
        theme_list,
        reviews,
        max_themes=max_themes,
    )
    payload["meta"]["method"] = "seeded themes + embedding assignment"
    if source_path:
        payload["meta"]["source"] = source_path
    return payload, points


def run_hybrid_clustering(
    raw_data: List[Dict],
    source_path: Optional[str] = None,
    max_themes: int = MAX_THEMES,
    allow_seed_fallback: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Full hybrid pipeline: sample → LLM themes → embed-assign → scatter payload."""
    from src.llm.assemble_prompts import build_review_theme_prompt
    from src.llm.query_llm import query_llm

    reviews = flatten_reviews(raw_data)
    sample_lines = build_llm_sample(reviews)
    print(f"LLM sample: {len(sample_lines)} truncated reviews "
          f"(from {len(reviews)} total)")

    try:
        prompt = build_review_theme_prompt(sample_lines, max_themes=max_themes)
        print("Requesting theme labels from LLM...")
        raw_response = query_llm(prompt_override=prompt, return_raw=True)
        themes = parse_theme_response(raw_response)
        print(f"LLM returned {len(themes)} theme labels")
        payload, points = assign_and_score(
            themes,
            reviews,
            max_themes=max_themes,
        )
        if source_path:
            payload["meta"]["source"] = source_path
        payload["meta"]["llm_sample_size"] = len(sample_lines)
        return payload, points
    except Exception as e:
        if not allow_seed_fallback:
            raise
        print(f"LLM theme extraction failed ({e}); falling back to seeded themes")
        payload, points = run_seeded_clustering(
            raw_data,
            source_path=source_path,
            max_themes=max_themes,
        )
        payload["meta"]["llm_fallback"] = str(e)
        payload["meta"]["llm_sample_size"] = len(sample_lines)
        return payload, points
