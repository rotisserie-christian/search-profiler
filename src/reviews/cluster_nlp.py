"""
Local NLP clustering for Google Maps reviews (no LLM / no API credits).

Pipeline:
  1. Split reviews into sentences, drop generic praise fluff
  2. Embed with MiniLM
  3. Cluster low-star and high-star sentences separately (agglomerative)
  4. Label each cluster with c-TF-IDF style phrases (sklearn)
  5. Score for dual scatter views (pain + market)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DISTANCE_THRESHOLD = 0.48
LOW_DISTANCE_THRESHOLD = 0.42
HIGH_DISTANCE_THRESHOLD = 0.58
MIN_SENTENCE_CHARS = 25
MIN_CLUSTER_SIZE = 4
MIN_UNIQUE_REVIEWS = 2
MAX_EXAMPLES = 3
MIN_PREVALENCE = 0.5
MAX_THEMES = 35
LOW_RATING_MAX = 2.0
HIGH_RATING_MIN = 4.0
LABEL_MERGE_SIM = 0.85


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|(?<=\n)\s*")
GENERIC_PRAISE = re.compile(
    r"\b("
    r"highly recommend|recommend(ed|ing)? (this|them|him|her)|"
    r"excellent service|great (job|service|work|experience|company)|"
    r"amazing (job|work|service|company)|fantastic|wonderful|"
    r"outstanding (job|work|service)|pleasure to work|"
    r"best (company|landscaper|contractor)|five stars|5 stars|"
    r"thank you so much|couldn'?t be happier"
    r")\b",
    re.I,
)

# Generic rant / warning language without a concrete operational issue
JUNK_RANT = re.compile(
    r"\b("
    r"stay away|steering clear|do not (hire|trust)|don'?t (hire|trust)|"
    r"would not recommend|do not recommend|don'?t recommend|"
    r"worst company|horrible company|terrible company|"
    r"bad experience|deeply disappointing|regret (using|hiring)"
    r")\b",
    re.I,
)

OPERATIONAL_CUES = re.compile(
    r"\b("
    r"callback|call back|voicemail|phone|email|communicat|"
    r"deadline|schedule|on time|late|delay|no[- ]show|show up|"
    r"snow|driveway|damage|deposit|price|quote|permit|grading|"
    r"quality|workmanship|cleanup|debris|design|budget|unfinished|"
    r"abandoned|incomplete|rude|unprofessional|warranty|refund|"
    r"plant|garden|hardscape|retaining|crew|project manager"
    r")\b",
    re.I,
)


def flatten_reviews(raw_data: List[Dict]) -> List[Dict]:
    """Extract review records with stable ids from a raw SerpApi fetch."""
    reviews = []
    for batch in raw_data:
        business = batch.get("business_info", {}) or {}
        business_name = business.get("title", "Unknown")
        for i, review in enumerate(batch.get("reviews", [])):
            snippet = (review.get("snippet") or "").strip()
            if not snippet:
                continue
            review_id = review.get("review_id") or f"{business_name}:{i}"
            reviews.append({
                "review_id": review_id,
                "rating": float(review.get("rating") or 0),
                "snippet": snippet,
                "business": business_name,
            })
    return reviews


def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT.split(text)
    sentences = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" \t\r\n-•*")
        if len(cleaned) < MIN_SENTENCE_CHARS:
            continue
        words = cleaned.split()
        if len(words) < 5:
            continue
        sentences.append(cleaned)
    return sentences


def build_sentence_units(reviews: List[Dict], drop_generic_praise: bool = True) -> List[Dict]:
    units = []
    for review in reviews:
        for sentence in split_sentences(review["snippet"]):
            if drop_generic_praise and _is_generic_praise(sentence, review["rating"]):
                continue
            units.append({
                "text": sentence,
                "rating": review["rating"],
                "review_id": review["review_id"],
                "business": review["business"],
            })
    return units


def _is_generic_praise(text: str, rating: float) -> bool:
    if rating < HIGH_RATING_MIN:
        return False
    if not GENERIC_PRAISE.search(text):
        return False
    # Keep longer, specific praise; drop short glowing fluff
    return len(text.split()) < 22


def _cluster_type(avg_rating: float) -> str:
    if avg_rating >= 4.0:
        return "Strength"
    if avg_rating <= 2.0:
        return "Weakness"
    return "Mixed"


def _medoid_index(embeddings: np.ndarray, member_idxs: List[int]) -> int:
    member_emb = embeddings[member_idxs]
    centroid = member_emb.mean(axis=0)
    norms = np.linalg.norm(member_emb, axis=1) * (np.linalg.norm(centroid) + 1e-9)
    sims = (member_emb @ centroid) / norms
    return member_idxs[int(np.argmax(sims))]


def _clean_label(phrase: str) -> str:
    phrase = re.sub(r"\s+", " ", phrase).strip(" .-_")
    # Light title case without uppercasing tiny words awkwardly
    return phrase[:1].upper() + phrase[1:] if phrase else phrase


def _phrase_label(
    member_texts: List[str],
    member_embeddings: np.ndarray,
    model: SentenceTransformer,
) -> Optional[str]:
    """
    Local KeyBERT-style label: candidate n-grams whose embedding is closest
    to the cluster centroid.
    """
    if not member_texts:
        return None

    banned = {
        "highly recommend", "great job", "great work", "great service",
        "excellent work", "amazing work", "thank you", "years ago",
        "would recommend", "highly recommended", "great experience",
        "steering clear", "looking landscaping",
    }

    candidates: List[str] = []
    for min_df in (2, 1):
        try:
            vec = TfidfVectorizer(
                ngram_range=(2, 3),
                stop_words="english",
                min_df=min_df,
                max_features=200,
            )
            vec.fit(member_texts)
            candidates = list(vec.get_feature_names_out())
            if candidates:
                break
        except ValueError:
            continue

    if not candidates:
        return None

    # Keep readable candidates only
    filtered = []
    for phrase in candidates:
        if phrase in banned:
            continue
        if any(ch.isdigit() for ch in phrase):
            continue
        words = phrase.split()
        if len(words) < 2 or len(words) > 3:
            continue
        # Drop fragments that look like broken grammar crumbs
        if words[0] in {"did", "does", "looks", "looking", "went", "come"}:
            continue
        filtered.append(phrase)
    if not filtered:
        filtered = [c for c in candidates if c not in banned][:40]
    if not filtered:
        return None

    centroid = member_embeddings.mean(axis=0)
    cand_emb = np.asarray(model.encode(filtered, show_progress_bar=False))
    c_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
    e_norm = cand_emb / (np.linalg.norm(cand_emb, axis=1, keepdims=True) + 1e-9)
    sims = e_norm @ c_norm
    best = filtered[int(np.argmax(sims))]
    return _clean_label(best)


def _cluster_band(
    indices: List[int],
    embeddings: np.ndarray,
    distance_threshold: float,
    min_cluster_size: int,
) -> Dict[int, List[int]]:
    if len(indices) < min_cluster_size:
        return {}
    sub = embeddings[indices]
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    local_labels = clustering.fit_predict(sub)
    groups: Dict[int, List[int]] = defaultdict(list)
    for local_i, lab in enumerate(local_labels):
        groups[int(lab)].append(indices[local_i])
    return {
        lab: members
        for lab, members in groups.items()
        if len(members) >= min_cluster_size
    }


def _merge_similar_points(
    points: List[Dict],
    model: SentenceTransformer,
    sim_threshold: float = LABEL_MERGE_SIM,
) -> List[Dict]:
    if len(points) <= 1:
        return points
    labels = [p["label"] for p in points]
    emb = np.asarray(model.encode(labels, show_progress_bar=False))
    norms = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = norms @ norms.T

    used = set()
    merged = []
    for i, p in enumerate(points):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(points)):
            if j in used:
                continue
            if float(sims[i, j]) >= sim_threshold and p["type"] == points[j]["type"]:
                group.append(j)
                used.add(j)
        used.add(i)
        best = max(
            group,
            key=lambda k: (
                points[k].get("prevalence_among_low", 0),
                points[k].get("prevalence_all", 0),
            ),
        )
        keep = dict(points[best])
        if len(group) > 1:
            keep["sentence_count"] = sum(points[k].get("sentence_count", 0) for k in group)
            keep["review_count"] = max(points[k].get("review_count", 0) for k in group)
            keep["low_review_count"] = max(
                points[k].get("low_review_count", 0) for k in group
            )
            keep["prevalence_all"] = max(
                points[k].get("prevalence_all", 0) for k in group
            )
            keep["prevalence_among_low"] = max(
                points[k].get("prevalence_among_low", 0) for k in group
            )
            keep["prevalence"] = keep.get("prevalence_all", keep.get("prevalence", 0))
            keep["y"] = keep["prevalence"]
            examples = []
            seen = set()
            for k in group:
                for ex in points[k].get("examples") or []:
                    if ex in seen:
                        continue
                    examples.append(ex)
                    seen.add(ex)
                    if len(examples) >= MAX_EXAMPLES:
                        break
                if len(examples) >= MAX_EXAMPLES:
                    break
            keep["examples"] = examples
        merged.append(keep)
    return merged


def is_junk_cluster(point: Dict) -> bool:
    """
    True when examples/label are generic rants/warnings with no operational cue.
    """
    blob = " ".join(
        [str(point.get("label") or "")] + list(point.get("examples") or [])
    )
    if not JUNK_RANT.search(blob):
        return False
    return OPERATIONAL_CUES.search(blob) is None


def filter_junk_clusters(points: List[Dict]) -> List[Dict]:
    kept = [p for p in points if not is_junk_cluster(p)]
    dropped = len(points) - len(kept)
    if dropped:
        print(f"Dropped {dropped} junk clusters (generic rant / no operational detail)")
    return kept


def cluster_review_feedback(
    raw_data: List[Dict],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    model_name: str = MODEL_NAME,
    max_themes: int = MAX_THEMES,
    low_rating_max: float = LOW_RATING_MAX,
) -> Tuple[Dict, List[Dict]]:
    """
    Discover themes locally and return a dual-view scatter payload.
    Primary `points` = pain view; `market.points` = full map.
    """
    reviews = flatten_reviews(raw_data)
    units = build_sentence_units(reviews, drop_generic_praise=True)
    total_reviews = len(reviews)
    low_review_ids = {r["review_id"] for r in reviews if r["rating"] <= low_rating_max}
    n_low = len(low_review_ids)
    all_texts = [u["text"] for u in units]

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
            "label": f"Prevalence among <={low_rating_max:.0f}-star reviews (%)",
            "domain": [0, 100],
        },
    }

    empty = {
        "meta": {
            "method": "local nlp: banded agglomerative + tfidf labels",
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
    if not units:
        return empty, []

    print(f"Loading {model_name} (local)...")
    model = SentenceTransformer(model_name)
    embeddings = np.asarray(model.encode(all_texts, show_progress_bar=True))

    low_idx = [i for i, u in enumerate(units) if u["rating"] <= low_rating_max]
    high_idx = [i for i, u in enumerate(units) if u["rating"] >= HIGH_RATING_MIN]

    low_thresh = min(distance_threshold, LOW_DISTANCE_THRESHOLD)
    high_thresh = max(distance_threshold, HIGH_DISTANCE_THRESHOLD)
    print(
        f"Clustering bands: low={len(low_idx)} sentences (t={low_thresh}), "
        f"high={len(high_idx)} sentences (t={high_thresh})"
    )
    low_groups = _cluster_band(low_idx, embeddings, low_thresh, max(3, min_cluster_size - 1))
    high_groups = _cluster_band(high_idx, embeddings, high_thresh, min_cluster_size)

    print(f"Raw clusters: {len(low_groups)} low-band + {len(high_groups)} high-band")

    def _build_point(member_idxs: List[int], band: str) -> Optional[Dict]:
        unique_reviews = {units[i]["review_id"] for i in member_idxs}
        if len(unique_reviews) < MIN_UNIQUE_REVIEWS:
            return None

        ratings = [units[i]["rating"] for i in member_idxs]
        avg_score = float(np.mean(ratings))
        prevalence_all = (
            round((len(unique_reviews) / total_reviews) * 100, 1) if total_reviews else 0.0
        )
        low_hits = unique_reviews & low_review_ids
        prevalence_low = round((len(low_hits) / n_low) * 100, 1) if n_low else 0.0
        if prevalence_all < MIN_PREVALENCE and prevalence_low < MIN_PREVALENCE:
            return None

        member_texts = [units[i]["text"] for i in member_idxs]
        member_emb = embeddings[member_idxs]
        label = _phrase_label(member_texts, member_emb, model)
        if not label:
            medoid_i = _medoid_index(embeddings, member_idxs)
            label = units[medoid_i]["text"]
            if len(label) > 60:
                label = label[:59].rsplit(" ", 1)[0] + "…"

        theme_type = _cluster_type(avg_score)
        # Low-band clusters are pain-oriented even if a couple mid ratings sneak in
        if band == "low" and theme_type == "Mixed" and avg_score <= 2.5:
            theme_type = "Weakness"
        if band == "low" and avg_score <= 2.0:
            theme_type = "Weakness"

        severity = round(max(0.0, 3.0 - avg_score), 2)
        ranked = sorted(member_idxs, key=lambda i: abs(units[i]["rating"] - avg_score))
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

        return {
            "id": 0,
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
            "sentence_count": len(member_idxs),
            "review_count": len(unique_reviews),
            "examples": examples,
            "_band": band,
        }

    low_points = []
    for members in low_groups.values():
        pt = _build_point(members, "low")
        if pt:
            low_points.append(pt)
    high_points = []
    for members in high_groups.values():
        pt = _build_point(members, "high")
        if pt:
            high_points.append(pt)

    low_points = filter_junk_clusters(low_points)
    high_points = filter_junk_clusters(high_points)
    low_points = _merge_similar_points(low_points, model)
    high_points = _merge_similar_points(high_points, model)
    low_points.sort(key=lambda p: (-p["prevalence_among_low"], p["score"]))
    high_points.sort(key=lambda p: (-p["prevalence_all"], p["score"]))

    # Reserve slots so praise volume cannot erase discovered pain themes
    pain_slots = max(10, max_themes // 3)
    strength_slots = max_themes - min(pain_slots, len(low_points))
    market_points = low_points[:pain_slots] + high_points[:strength_slots]
    market_points = filter_junk_clusters(market_points)
    market_points = _merge_similar_points(market_points, model, sim_threshold=LABEL_MERGE_SIM)
    market_points.sort(key=lambda p: (-p["prevalence_all"], p["score"]))
    for i, p in enumerate(market_points):
        p["id"] = i
        p.pop("_band", None)

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
    for i, p in enumerate(pain_points):
        p["id"] = i

    payload = {
        "meta": {
            "method": "local nlp: banded agglomerative + tfidf labels",
            "primary_view": "pain",
            "views": {
                "pain": "Weakness themes; y = share of low-star reviews",
                "market": "All themes; y = share of all fetched reviews",
            },
            "embedding_model": model_name,
            "distance_threshold": distance_threshold,
            "min_cluster_size": min_cluster_size,
            "total_reviews": total_reviews,
            "low_reviews": n_low,
            "low_rating_max": low_rating_max,
            "total_sentences": len(units),
            "clusters_kept": len(pain_points),
            "market_clusters": len(market_points),
            "max_themes": max_themes,
            "credits_used": 0,
        },
        "axes": pain_axes,
        "points": pain_points,
        "market": {
            "axes": market_axes,
            "points": market_points,
        },
    }
    return payload, pain_points


def cluster_from_raw_path(
    path: str,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> Dict:
    import json

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    payload, _ = cluster_review_feedback(raw_data, distance_threshold=distance_threshold)
    payload["meta"]["source"] = path
    return payload


def _parse_title_response(raw: str) -> Dict[int, Dict]:
    """Parse LLM title JSON into {id: {feedback, type}}."""
    import json

    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Title response was not a JSON list")

    out: Dict[int, Dict] = {}
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        label = (item.get("feedback") or item.get("label") or "").strip()
        if not label:
            continue
        hinted = (item.get("type") or "").strip().title()
        if hinted == "Discard" or label.lower() == "discard":
            out[int(item["id"])] = {"feedback": "Discard", "type": "Discard"}
            continue
        if hinted not in ("Strength", "Weakness", "Mixed"):
            hinted = None
        out[int(item["id"])] = {"feedback": label, "type": hinted}
    return out


def apply_title_map(payload: Dict, title_map: Dict[int, Dict]) -> Dict:
    """
    Apply a title map (from LLM or manual stand-in), drop Discard, merge near-dupes.
    """
    market_points = list((payload.get("market") or {}).get("points") or payload.get("points") or [])
    if not market_points:
        return payload

    kept = []
    discarded = 0
    for p in market_points:
        update = title_map.get(int(p["id"]))
        if update and update.get("type") == "Discard":
            discarded += 1
            continue
        if update:
            p = dict(p)
            p["label"] = update["feedback"]
            p["feedback"] = update["feedback"]
            if update.get("type"):
                p["type"] = update["type"]
        # Post-title junk check (vague banlist titles)
        if is_junk_cluster(p):
            discarded += 1
            continue
        kept.append(p)

    if discarded:
        print(f"Discarded {discarded} clusters after title pass")

    print("Merging near-duplicate titles...")
    model = SentenceTransformer(MODEL_NAME)
    kept = _merge_similar_points(kept, model, sim_threshold=0.80)
    kept.sort(key=lambda p: (-p.get("prevalence_all", 0), p.get("score", 0)))
    for i, p in enumerate(kept):
        p["id"] = i
        # Refresh scatter y for market view
        p["y"] = p.get("prevalence_all", p.get("y", 0))
        p["prevalence"] = p["y"]

    if payload.get("market") is not None:
        payload["market"]["points"] = kept
        payload["points"] = _rebuild_pain_points(kept)
    else:
        payload["points"] = kept

    meta = payload.setdefault("meta", {})
    meta["clusters_kept"] = len(payload.get("points") or [])
    meta["market_clusters"] = len(kept)
    return payload


def _rebuild_pain_points(market_points: List[Dict]) -> List[Dict]:
    pain = []
    for p in market_points:
        if p["type"] == "Weakness" or (p["type"] == "Mixed" and p.get("score", 5) <= 2.5):
            if p.get("low_review_count", 0) < MIN_UNIQUE_REVIEWS:
                continue
            pain_p = dict(p)
            pain_p["y"] = p.get("prevalence_among_low", p.get("y", 0))
            pain_p["prevalence"] = pain_p["y"]
            pain.append(pain_p)
    pain.sort(key=lambda p: (-p["prevalence"], p.get("score", 0)))
    for i, p in enumerate(pain):
        p["id"] = i
    return pain


def retitle_payload_with_llm(payload: Dict) -> Dict:
    """
    Replicate pass: rename clusters using 3–5 examples each.
    Does not re-cluster; score/prevalence stay local.
    """
    from src.llm.assemble_prompts import build_cluster_title_prompt
    from src.llm.query_llm import query_llm

    market_points = (payload.get("market") or {}).get("points") or []
    if not market_points:
        market_points = payload.get("points") or []
    if not market_points:
        return payload

    # Hygiene before spending tokens
    market_points = filter_junk_clusters(market_points)
    for i, p in enumerate(market_points):
        p["id"] = i
    if payload.get("market") is not None:
        payload["market"]["points"] = market_points
    else:
        payload["points"] = market_points

    clusters_for_prompt = [
        {
            "id": p["id"],
            "label": p.get("label") or p.get("feedback") or "",
            "type": p.get("type") or "",
            "examples": p.get("examples") or [],
        }
        for p in market_points
    ]
    prompt = build_cluster_title_prompt(clusters_for_prompt)
    print(f"Requesting LLM titles for {len(clusters_for_prompt)} clusters via Replicate...")
    raw = query_llm(prompt_override=prompt, return_raw=True)
    title_map = _parse_title_response(raw)
    print(f"LLM returned titles for {len(title_map)} clusters")

    payload = apply_title_map(payload, title_map)
    meta = payload.setdefault("meta", {})
    meta["method"] = "local nlp clusters + replicate titles"
    meta["title_pass"] = "replicate"
    meta["credits_used"] = meta.get("credits_used", 0)
    return payload
