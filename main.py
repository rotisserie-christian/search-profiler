import argparse
from pathlib import Path

from src.llm.query_llm import query_llm
from src.llm.consolidate import generate_consolidated_clusters
from src.llm.save_output import (
    save_queries_to_csv,
    save_clustered_queries_to_csv,
    save_clustered_queries_to_txt,
    save_clustered_queries_to_json,
)
from src.validate import run_validation
from src.merge import run_merge
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, validate, and manage search queries based on a user profile."
    )

    # Generation
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of LLM runs with semantic consolidation (default: 1)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cluster similarity threshold when using --runs (0-1, default: 0.75)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["head", "tail", "both"],
        default="both",
        help="Heuristic bias for generation: 'head' (short, high-volume terms "
             "for Trends), 'tail' (long-tail persona terms), or 'both' (default)"
    )

    # Validation
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="Validate queries from a searchtermsN.json file using Google Trends"
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="free music maker",
        help="Anchor term for cross-batch normalization (default: 'free music maker')"
    )

    # Explore
    parser.add_argument(
        "--explore",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="Fetch related queries for terms in a searchtermsN.json file"
    )

    # Joyplot
    parser.add_argument(
        "--joyplot",
        type=str,
        default=None,
        metavar="CSV_FILE",
        help="Fetch a single multi-term interest-over-time series from a CSV of terms (max 5)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default="today 5-y",
        help="Google Trends date range for --joyplot (default: 'today 5-y')"
    )

    # Merge
    parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("SEARCHTERMS_JSON", "VALIDATED_JSON"),
        help="Merge new queries from a searchterms file into an existing validated file"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip Google Trends validation when merging (default: validate)"
    )

    # Reviews
    parser.add_argument(
        "--reviews",
        action="store_true",
        help="Run the Google Maps reviews pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query for the reviews pipeline (e.g. 'Best cafes in Brooklyn')"
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Business type filter for reviews (e.g. 'Coffee shop')"
    )
    parser.add_argument(
        "--load-raw",
        type=str,
        default=None,
        help="Path to a previously fetched raw_reviews JSON file — skips live fetch"
    )
    parser.add_argument(
        "--cluster-method",
        type=str,
        choices=["nlp-llm", "nlp", "hybrid", "seeded", "llm"],
        default="nlp",
        help="Reviews clustering: 'nlp' (local only, default), "
             "'nlp-llm' (local clusters + Replicate titles), "
             "'seeded' (fixed landscaping themes), "
             "'hybrid' (LLM proposes themes then local assign), "
             "or 'llm' (legacy one-shot)"
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.48,
        help="Cosine distance threshold for NLP review clustering (default: 0.48)"
    )
    parser.add_argument(
        "--max-themes",
        type=int,
        default=35,
        help="Max theme clusters for hybrid reviews output (default: 35)"
    )

    args = parser.parse_args()

    # Validate
    if args.validate:
        input_path = Path(args.validate)

        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            logger.info("Generate queries first with: python main.py --runs 3")
            return

        logger.info("=" * 60)
        logger.info(f"VALIDATE: {input_path}")
        if args.anchor:
            logger.info(f"Anchor term: {args.anchor}")
        logger.info("=" * 60)

        try:
            validated_path = run_validation(input_path, anchor=args.anchor)
            if validated_path:
                logger.info("=" * 60)
                logger.info(f"Results saved to: {validated_path}")
                logger.info("=" * 60)
        except Exception:
            logger.exception("Validation failed")

        return

    # Explore
    if args.explore:
        input_path = Path(args.explore)

        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return

        logger.info("=" * 60)
        logger.info(f"EXPLORE: {input_path}")
        logger.info("=" * 60)

        try:
            from src.explore import run_explore
            updated_path = run_explore(input_path)
            if updated_path:
                logger.info("=" * 60)
                logger.info(f"Updated file: {updated_path}")
                logger.info("=" * 60)
        except Exception:
            logger.exception("Exploration failed")

        return

    # Joyplot
    if args.joyplot:
        input_path = Path(args.joyplot)

        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return

        logger.info("=" * 60)
        logger.info(f"JOYPLOT: {input_path} (date={args.date})")
        logger.info("=" * 60)

        try:
            from src.joyplot import run_joyplot
            output_path = run_joyplot(input_path, date=args.date)
            if output_path:
                logger.info("=" * 60)
                logger.info(f"Results saved to: {output_path}")
                logger.info("=" * 60)
        except Exception:
            logger.exception("Joyplot failed")

        return

    # Merge
    if args.merge:
        search_path, validated_path = args.merge

        logger.info("=" * 60)
        logger.info(f"MERGE: {search_path} → {validated_path}")
        logger.info("=" * 60)

        try:
            output_path = run_merge(
                search_path,
                validated_path,
                validate_new=not args.no_validate,
            )
            if output_path:
                logger.info("=" * 60)
                logger.info(f"Results saved to: {output_path}")
                logger.info("=" * 60)
        except Exception:
            logger.exception("Merge failed")

        return

    # Reviews
    if args.reviews:
        if not args.query and not args.load_raw:
            logger.error("--reviews requires --query or --load-raw")
            return

        try:
            from src.reviews import run_reviews
            run_reviews(
                args.query,
                args.type,
                load_raw_path=args.load_raw,
                method=args.cluster_method,
                distance_threshold=args.distance_threshold,
                max_themes=args.max_themes,
            )
        except Exception:
            logger.exception("Reviews pipeline failed")

        return

    # Generate
    if args.runs > 1:
        logger.info(f"Running LLM {args.runs} times (threshold={args.threshold}, mode={args.mode})...")

        clustered_queries = generate_consolidated_clusters(args.runs, args.threshold, mode=args.mode)

        csv_path = save_clustered_queries_to_csv(clustered_queries)
        txt_path = save_clustered_queries_to_txt(clustered_queries)
        json_path = save_clustered_queries_to_json(clustered_queries)

        logger.info(f"Generated {len(clustered_queries)} queries.")
        logger.info(f"Saved to:\n  - {csv_path}\n  - {txt_path}\n  - {json_path}")
    else:
        queries = query_llm(mode=args.mode)
        path = save_queries_to_csv(queries)

        logger.info(f"Generated {len(queries)} queries.")
        logger.info(f"Saved to: {path}")


if __name__ == "__main__":
    main()