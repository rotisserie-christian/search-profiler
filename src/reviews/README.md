### /reviews
- **`fetch_places.py`** - Extracts the top businesses and their IDs from Google, `--type` filtering
- **`fetch_reviews.py`** - Pulls paginated review data for a set of business IDs
- **`processor.py`** - Cleans text data and correlates star ratings to semantic clusters
- **`cluster_nlp.py`** - Local sentence clustering + phrase labels; junk filter; optional Replicate title pass
- **`cluster_hybrid.py`** - Seeded / sample-LLM theme lists + embedding assignment
- **`orchestrator.py`** - Main entry point

### Quality pipeline (`--cluster-method nlp-llm`)

```
full review dump
  → local NLP: sentence split + cluster (entire set)  
  → filter junk / merge near-duplicate clusters         
  → for each cluster: send 3–5 example sentences
  → LLM: actionable 2–5 word title (+ Strength/Weakness/Discard)
  → merge similar titles + drop Discard                 
  → keep local score / prevalence (already computed)
  → scatter JSON
```

Local-only discovery/scoring: `--cluster-method nlp` (default).  
Title polish via Replicate: `--cluster-method nlp-llm`.
