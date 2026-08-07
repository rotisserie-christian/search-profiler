# Semantic Maps

This is my market research toolkit, it can be used for a few different things:

#### Brainstorm
Generate a preliminary list of keywords based on the search behaviour of a given user persona

#### Validation 
Validates a set of keywords against Google Trends

#### Joyplot
Pulls 5 years of Google Trends interest data for up to 5 terms

#### Reviews
Pulls Google reviews from businesses in a given market, semantically clusters the feedback, and sorts them by ratings

## Contents
- [Set up](#set-up)
- [Keyword Brainstorming](#keyword-brainstorming)
  - [Multiple Runs](#multiple-runs)
  - [Manual Pruning](#manual-pruning)
  - [Manual Addition](#manual-addition)
  - [Explore Related Terms](#explore-related-terms)
- [Trends Validation](#trends-validation)
  - [Validation](#validation)
  - [Normalization (Anchor Terms)](#normalization-anchor-terms)
- [Joyplot](#joyplot)
- [Google Maps Reviews](#google-maps-reviews)
  - [Live Fetch](#live-fetch)
  - [Cached Replay](#cached-replay)
- [Dependencies](#dependencies)

## Set up

Clone the repo:

```bash
git clone https://github.com/rotisserie-christian/search-profiler
```

Install dependencies (Python 3.10):

```bash
pip install -r requirements.txt
```

Add your **`REPLICATE_API_TOKEN`** and **`SERPAPI_API_KEY`**, fill out the user profile in **`src/config.py`**

## Keyword Brainstorming

### Multiple runs 

This will query the LLM x times, collect all unique search terms, and consolidate the output

> [!TIP]
> Adjust max_tokens in src/config.py to limit the number of generated terms. Usually ~100 tokens = 1 search term.

```bash
python main.py --runs x
```

This also consolidates the cluster titles based on semantic similarity

Adjust the similarity threshold (default 0.75):

```bash
python main.py --runs x --threshold y
```

> [!WARNING]  
> This creates a lot more output tokens. This is why I set the default model to Deepseek. A high number of runs with a more expensive model can create a massive bill very quickly. 

### Head vs long-tail (`--mode`)

Google Trends only has reliable volume for short, head-style queries. As of right now, there is no system here for validating long-tail terms, but you can still generate them if you want to. The `--mode` flag controls which heuristics drive generation:

- **`head`** - short, high-volume terms (uses `word_choice`, `reformulation`); best for Trends validation
- **`tail`** - long-tail persona terms (uses `query_length`, `complexity`, `specificity`); best for intent mapping / SEO
- **`both`** - all heuristics (default)

```bash
python main.py --runs x --mode head
```

### Manual pruning 

I would recommend pruning any slop generations from the JSON output before validating. This will conserve API credits. 

This needs to be done to the JSON file in particular, since this is the format used to call SerpAPI. The TXT and CSV outputs are meant for quick readability and export, and are not used in the actual script itself. 

### Manual addition

You can also add your own queries to the JSON file

cd into **`src/utils`** and run:

```bash
python add_query.py searchtermsN.json
```

This will check if it exists, if it doesn't, it will use `sentence-transformers` to find the best matching cluster and add the query to it. 

### Explore related terms 

> [!WARNING]  
> This step can use a lot of serpAPI credits. It also tends to return queries that are less relevant to the specific search intent of the given user profile. However, it can sometimes return highly valuable queries. Just be aware that this is an optional step with a slot machine mechanic baked into it. 

This will call serpAPI to retrieve related queries for each search term if they exist, and add them to the appropriate cluster. It takes in a searchtermsN.json file as an argument and writes the new terms to the same file.

```bash
python main.py --explore output/searchtermsN.json
```

## Trends Validation

### Validation

Run the **`--validate`** flag to call SerpAPI, creating a new JSON file containing search interest data for each term. It will omit any terms with 0 data and write the result to `/output/validatedtermsN.json`

```bash
python main.py --validate output/searchtermsN.json --anchor "your anchor term"
```

> [!NOTE]  
> It has to be the JSON file, not the CSV or TXT file.

### Normalization (Anchor Terms)

Google Trends data is relative (0-100) and specific to the terms in a single query. To compare hundreds of terms across different batches, you **must** use an anchor term.

By passing the `--anchor` flag, the script:
1. Includes your anchor in every API batch.
2. Calculates a "Batch Multiplier" based on the anchor's performance.
3. Rebases all other terms in that batch against a global reference scale.

**Without an anchor, high-volume and low-volume terms will look identical on a chart if they are in different batches.**

## Joyplot

Fetch 5 years of Google Trends interest data for up to 5 terms and save the time series to `/output/joyplot/joyplotdataN.json`

It takes a CSV of terms (one per line, max 5) and makes a **single** SerpAPI call:

```bash
python main.py --joyplot output/joyplottermsN.csv
```

Override the default 5-year window (`today 5-y`):

```bash
python main.py --joyplot output/joyplottermsN.csv --date "today 12-m"
```

> [!NOTE]
> This uses one API credit. Because it only uses 5 terms, no anchor or multiple requests are needed  

## Google Maps Reviews

Cluster business reviews from Google Maps to surface recurring feedback themes for a scatterplot (**x** = average star score, **y** = prevalence %).

### Recommended quality pipeline (`nlp-llm`)

Coverage stays local over the **entire** review dump; Replicate is used only to name clusters:

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

Junk clusters (generic “stay away” / “would not recommend” with no operational detail) are dropped before titling when possible. The LLM may also mark a cluster `Discard` if examples cannot support a concrete theme.

```bash
python main.py --reviews --load-raw output/raw_reviews/latest_fetch.json --cluster-method nlp-llm
```

Needs `REPLICATE_API_TOKEN` and Replicate credit. Does **not** re-fetch reviews when using `--load-raw`.

### Local only (`nlp`, default)

Same clustering/scoring, automatic phrase labels (no Replicate):

```bash
python main.py --reviews --load-raw output/raw_reviews/latest_fetch.json --cluster-method nlp
```

### Other methods
- **`seeded`** — fixed landscaping theme list + local assignment  
- **`hybrid`** — LLM proposes themes from a sample, then local assignment (fallback to seeded)  
- **`llm`** — legacy one-shot Replicate over all review text  

### Live Fetch
```bash
python main.py --reviews --query "<business> in <location>" --type "<business_type>"
```

### Output
`output/reviews/review_scatterN.json` (dual-view):
- **`points`** (primary / pain): weakness themes; **y** = prevalence among ≤2★ reviews  
- **`market.points`**: all themes; **y** = prevalence among all reviews  

Optional: `--max-themes 25`, `--distance-threshold 0.48`.

## Dependencies 
- **`Replicate`** - LLM API
- **`sentence-transformers`** - semantic clustering
- **`scikit-learn`** - cosine similarity
- **`numpy`** - numerical operations
- **`requests`** - HTTP requests
