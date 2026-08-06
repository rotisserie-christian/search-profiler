import json
from .fetch_places import fetch_top_places
from .fetch_reviews import fetch_reviews_for_places
from .processor import prepare_reviews_for_llm, save_raw_reviews
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_reviews(
    query: str = None,
    business_type: str = None,
    load_raw_path: str = None,
    method: str = "hybrid",
    distance_threshold: float = 0.48,
    max_themes: int = 35,
):
    """
    Orchestrates the full reviews pipeline.

    method:
        "nlp-llm" — local NLP over full set, then Replicate titles only
        "nlp" — local sentence clustering + phrase labels (no Replicate)
        "hybrid" — LLM proposes themes from a sample, then local assignment
        "seeded" — fixed landscaping theme list + local assignment
        "llm" — legacy one-shot Replicate clustering
    """
    if load_raw_path:
        logger.info(f"Loading raw reviews from: {load_raw_path}")
        try:
            with open(load_raw_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            return
    else:
        if not query:
            logger.error("A --query is required unless --load-raw is provided.")
            return

        logger.info(f"Starting Reviews Pipeline for query: '{query}'")
        if business_type:
            logger.info(f"Filtering for business type: '{business_type}'")

        try:
            places = fetch_top_places(query, business_type)
            if not places:
                logger.warning("No places found matching criteria.")
                return
            logger.info(f"Found {len(places)} businesses.")
        except Exception as e:
            logger.error(f"Failed to fetch places: {e}")
            return

        try:
            raw_data = fetch_reviews_for_places(places, max_pages_per_side=2)
            if not raw_data:
                logger.warning("No reviews collected.")
                return

            save_raw_reviews(raw_data)
        except Exception as e:
            logger.error(f"Failed to fetch reviews: {e}")
            return

    if method == "nlp-llm":
        return _run_nlp_llm_clustering(raw_data, load_raw_path, distance_threshold, max_themes)
    if method == "hybrid":
        return _run_hybrid_clustering(raw_data, load_raw_path, max_themes)
    if method == "seeded":
        return _run_seeded_clustering(raw_data, load_raw_path, max_themes)
    if method == "nlp":
        return _run_nlp_clustering(raw_data, load_raw_path, distance_threshold, max_themes)
    return _run_llm_clustering(raw_data)


def _run_seeded_clustering(raw_data, source_path: str = None, max_themes: int = 35):
    try:
        from .cluster_hybrid import run_seeded_clustering
        from src.llm.save_output import save_review_scatter_to_json

        logger.info(f"Running seeded theme clustering (max_themes={max_themes})...")
        payload, points = run_seeded_clustering(
            raw_data,
            source_path=source_path,
            max_themes=max_themes,
        )
        output_path = save_review_scatter_to_json(payload)

        logger.info("-" * 40)
        logger.info("REVIEWS SEEDED ANALYSIS COMPLETE")
        logger.info(
            f"Pain clusters: {len(points)} | "
            f"market clusters: {payload['meta'].get('market_clusters', '?')} | "
            f"reviews={payload['meta']['total_reviews']} "
            f"(low={payload['meta'].get('low_reviews', '?')})"
        )
        logger.info(f"Scatterplot JSON: {output_path}")
        logger.info("Primary points: pain view (y = % of low-star reviews)")
        logger.info("-" * 40)
        return output_path
    except Exception as e:
        logger.error(f"Seeded clustering failed: {e}")
        logger.exception("Seeded error detail")
        return


def _run_hybrid_clustering(raw_data, source_path: str = None, max_themes: int = 35):
    try:
        from .cluster_hybrid import run_hybrid_clustering
        from src.llm.save_output import save_review_scatter_to_json

        logger.info(f"Running hybrid clustering (max_themes={max_themes})...")
        payload, points = run_hybrid_clustering(
            raw_data,
            source_path=source_path,
            max_themes=max_themes,
        )
        output_path = save_review_scatter_to_json(payload)

        logger.info("-" * 40)
        logger.info("REVIEWS HYBRID ANALYSIS COMPLETE")
        logger.info(
            f"Pain clusters: {len(points)} | "
            f"market clusters: {payload['meta'].get('market_clusters', '?')} | "
            f"reviews={payload['meta']['total_reviews']} "
            f"(low={payload['meta'].get('low_reviews', '?')})"
        )
        logger.info(f"Scatterplot JSON: {output_path}")
        logger.info("Primary points: pain view (y = % of low-star reviews)")
        logger.info("-" * 40)
        return output_path
    except Exception as e:
        logger.error(f"Hybrid clustering failed: {e}")
        logger.exception("Hybrid error detail")
        return


def _run_nlp_clustering(
    raw_data,
    source_path: str = None,
    distance_threshold: float = 0.48,
    max_themes: int = 35,
):
    try:
        from .cluster_nlp import cluster_review_feedback
        from src.llm.save_output import save_review_scatter_to_json

        logger.info(
            f"Running local NLP clustering (distance_threshold={distance_threshold}, no Replicate)..."
        )
        payload, points = cluster_review_feedback(
            raw_data,
            distance_threshold=distance_threshold,
            max_themes=max_themes,
        )
        if source_path:
            payload["meta"]["source"] = source_path

        output_path = save_review_scatter_to_json(payload)

        logger.info("-" * 40)
        logger.info("REVIEWS NLP ANALYSIS COMPLETE")
        logger.info(
            f"Pain clusters: {len(points)} | "
            f"market clusters: {payload['meta'].get('market_clusters', '?')} | "
            f"reviews={payload['meta']['total_reviews']} "
            f"(low={payload['meta'].get('low_reviews', '?')})"
        )
        logger.info(f"Scatterplot JSON: {output_path}")
        logger.info("Primary points: pain view (y = % of low-star reviews)")
        logger.info("-" * 40)
        return output_path
    except Exception as e:
        logger.error(f"NLP clustering failed: {e}")
        logger.exception("NLP error detail")
        return


def _run_nlp_llm_clustering(
    raw_data,
    source_path: str = None,
    distance_threshold: float = 0.48,
    max_themes: int = 35,
):
    """
    Full-set local NLP for coverage, then Replicate only to title clusters.
    """
    try:
        from .cluster_nlp import cluster_review_feedback, retitle_payload_with_llm
        from src.llm.save_output import save_review_scatter_to_json

        logger.info(
            f"Running NLP+LLM pipeline "
            f"(local cluster, Replicate titles; threshold={distance_threshold})..."
        )
        payload, _ = cluster_review_feedback(
            raw_data,
            distance_threshold=distance_threshold,
            max_themes=max_themes,
        )
        if source_path:
            payload["meta"]["source"] = source_path

        payload = retitle_payload_with_llm(payload)
        points = payload.get("points") or []
        output_path = save_review_scatter_to_json(payload)

        logger.info("-" * 40)
        logger.info("REVIEWS NLP+LLM ANALYSIS COMPLETE")
        logger.info(
            f"Pain clusters: {len(points)} | "
            f"market clusters: {payload['meta'].get('market_clusters', '?')} | "
            f"reviews={payload['meta']['total_reviews']} "
            f"(low={payload['meta'].get('low_reviews', '?')})"
        )
        logger.info(f"Scatterplot JSON: {output_path}")
        logger.info("Titles from Replicate; scores/prevalence from local NLP")
        logger.info("-" * 40)
        return output_path
    except Exception as e:
        logger.error(f"NLP+LLM clustering failed: {e}")
        logger.exception("NLP+LLM error detail")
        return


def _run_llm_clustering(raw_data):
    try:
        review_texts = prepare_reviews_for_llm(raw_data)
        logger.info(f"Prepared {len(review_texts)} review snippets for semantic analysis.")

        from src.llm.assemble_prompts import build_review_prompt
        from src.llm.query_llm import query_llm
        from src.llm.save_output import save_reviews_to_json
        from .processor import calculate_cluster_metrics

        prompt = build_review_prompt(review_texts)

        logger.info("Executing semantic clustering via LLM...")
        llm_response = query_llm(prompt_override=prompt, return_raw=True)

        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_response:
            llm_response = llm_response.split("```")[1].split("```")[0].strip()

        clusters = json.loads(llm_response)

        final_clusters = calculate_cluster_metrics(clusters, raw_data)
        output_path = save_reviews_to_json(final_clusters)

        logger.info("-" * 40)
        logger.info("REVIEWS ANALYSIS COMPLETE")
        logger.info(f"Final output: {output_path}")
        logger.info("-" * 40)
        return output_path

    except Exception as e:
        logger.error(f"LLM processing failed: {e}")
        return
