from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import csv
import json


def _next_output_path(output_dir: Path, extension: str = "csv") -> Path:
    """
    Return the next available output path, numbered sequentially
    
    Args:
        output_dir: Directory to save the file
        extension: File extension (without dot)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n = 1
    while True:
        candidate = output_dir / f"searchterms{n}.{extension}"
        if not candidate.exists():
            return candidate
        n += 1


def save_queries_to_csv(
    queries: Iterable[str],
    output_root: Path | str = "output",
) -> Path:
    """
    Write queries to a new CSV file 

    The file is named `searchtermsN.csv` where N is the first positive integer
    that doesn't already exist

    Returns:
        The Path of the CSV file that was written
    """
    output_dir = Path(output_root)
    output_path = _next_output_path(output_dir, "csv")

    # Normalize to a concrete list so it can be inspected
    query_list: List[str] = [q for q in queries if q]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query"])
        for q in query_list:
            writer.writerow([q])

    return output_path


def save_clustered_queries_to_csv(
    clustered_queries: Iterable[Tuple[str, str]],
    output_root: Path | str = "output",
) -> Path:
    """
    Write clustered queries to a new CSV file with cluster and query columns.
    
    Args:
        clustered_queries: Iterable of (cluster_title, query) tuples
        output_root: Directory to save the CSV file
        
    Returns:
        The Path of the CSV file that was written
    """
    output_dir = Path(output_root)
    output_path = _next_output_path(output_dir, "csv")
    
    # Normalize to a concrete list
    query_list: List[Tuple[str, str]] = [(c, q) for c, q in clustered_queries if c and q]
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "query"])
        for cluster, query in query_list:
            writer.writerow([cluster, query])
    
    return output_path


def save_clustered_queries_to_txt(
    clustered_queries: Iterable[Tuple[str, str]],
    output_root: Path | str = "output",
) -> Path:
    """
    Write clustered queries to a new TXT file with cluster titles as headers.
    
    Format:
        Cluster Title 1
        - query 1
        - query 2
        
        Cluster Title 2
        - query 3
        - query 4
    
    Args:
        clustered_queries: Iterable of (cluster_title, query) tuples
        output_root: Directory to save the TXT file
        
    Returns:
        The Path of the TXT file that was written
    """
    output_dir = Path(output_root)
    output_path = _next_output_path(output_dir, "txt")
    
    # Group queries by cluster
    clusters_dict: Dict[str, List[str]] = {}
    for cluster, query in clustered_queries:
        if cluster and query:
            if cluster not in clusters_dict:
                clusters_dict[cluster] = []
            clusters_dict[cluster].append(query)
    
    # Write to file
    with output_path.open("w", encoding="utf-8") as f:
        for i, (cluster_title, queries) in enumerate(clusters_dict.items()):
            # Add blank line between clusters (except before first)
            if i > 0:
                f.write("\n")
            
            # Write cluster title
            f.write(f"{cluster_title}\n")
            
            # Write queries
            for query in queries:
                f.write(f"- {query}\n")
    
    return output_path


def save_clustered_queries_to_json(
    clustered_queries: Iterable[Tuple[str, str]],
    output_root: Path | str = "output",
) -> Path:
    """
    Write clustered queries to a new JSON file.
    
    Format:
        [
            {
                "cluster": "Cluster Title 1",
                "queries": ["query 1", "query 2"]
            },
        ]
    
    Args:
        clustered_queries: Iterable of (cluster_title, query) tuples
        output_root: Directory to save the JSON file
        
    Returns:
        The Path of the JSON file that was written
    """
    output_dir = Path(output_root)
    output_path = _next_output_path(output_dir, "json")
    
    # Group queries by cluster
    clusters_dict: Dict[str, List[str]] = {}
    for cluster, query in clustered_queries:
        if cluster and query:
            if cluster not in clusters_dict:
                clusters_dict[cluster] = []
            clusters_dict[cluster].append(query)
    
    # Convert to list of dicts
    clusters_list = [
        {"cluster": cluster, "queries": queries}
        for cluster, queries in clusters_dict.items()
    ]
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(clusters_list, f, indent=2, ensure_ascii=False)
    
    return output_path


def save_reviews_to_json(
    clusters: List[Dict],
    output_root: Path | str = "output",
    prefix: str = "reviews",
) -> Path:
    """
    Write semantic review clusters to a new JSON file.
    """
    output_dir = Path(output_root) / "reviews"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple sequential naming
    n = 1
    while True:
        candidate = output_dir / f"{prefix}{n}.json"
        if not candidate.exists():
            output_path = candidate
            break
        n += 1
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    
    return output_path


def save_review_scatter_to_json(
    payload: Dict,
    output_root: Path | str = "output",
) -> Path:
    """
    Write scatterplot-ready review theme clusters (x=score, y=prevalence).
    """
    return save_reviews_to_json(payload, output_root=output_root, prefix="review_scatter")
