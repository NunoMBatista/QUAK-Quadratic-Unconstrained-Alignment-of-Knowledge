import re
from pathlib import Path
from typing import List, Optional, Tuple

import arxiv
import wikipedia

from src.config import (
    DOMAIN_QUERY,
    NUM_ARTICLES,
    QUERY_MODE,
    RAW_ARXIV_DIR,
    RAW_WIKI_DIR,
    WIKI_PAGE_TITLES,
    ARXIV_PAPER_IDS,
)


def _sanitize_filename(value: str, fallback: str) -> str:
    """return a filesystem-friendly version of a string."""
    cleaned = re.sub(r"\s+", "_", value.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _save_document(directory: Path, filename: str, content: str) -> Path:
    """save given document to /output/articles/[arxiv|wiki]"""
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / filename
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(content)
    return file_path


def _extract_arxiv_id(entry_id: str) -> str:
    """Return a usable arXiv identifier from an entry URL."""
    if "/abs/" in entry_id:
        return entry_id.split("/abs/")[-1]
    return entry_id.rsplit("/", 1)[-1]


def _load_cached_records(directory: Path, key_prefix: str) -> List[Tuple[str, str]]:
    """Return cached (key, body) pairs parsed from saved article files."""
    if not directory.exists():
        return []

    records: List[Tuple[str, str]] = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        header, sep, body = text.partition("---\n")
        key = None
        for line in header.splitlines():
            if line.startswith(key_prefix):
                key = line[len(key_prefix) :].strip()
                break
        if key is None:
            continue
        content = body.strip() if sep else text.strip()
        records.append((key, content))
    return records


def _next_file_index(directory: Path) -> int:
    """Determine the next numeric prefix for a saved document."""
    if not directory.exists():
        return 1
    max_idx = 0
    for path in directory.glob("*.txt"):
        stem = path.stem
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            max_idx = max(max_idx, int(prefix))
    return max_idx + 1


def fetch_wiki_data(titles: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """
    fetch a list of Wikipedia page summaries based on a query.
    
    this function can either search for the most relevant articles or 
    just fetch some predetermined ones. To define its behavior, go into
    config.py and read the section regarding to "entities and domain 
    configuration".
    
    Parameters
    ----------
    titles : Optional[list[str]]
        Optional explicit list of Wikipedia page titles to fetch. When provided,
        this list will be used instead of the configured `WIKI_PAGE_TITLES`.
        
    Returns
    -------
    tuple[list[str], list[str]]
        Summaries and their associated Wikipedia page titles.
    """

    cached_entries = _load_cached_records(RAW_WIKI_DIR, "Title: ")
    cached_map = {title: summary for title, summary in cached_entries}

    expected_titles: Optional[List[str]] = None
    # honor an explicit list of titles provided by the caller
    if titles is not None:
        expected_titles = titles[:NUM_ARTICLES]
    elif QUERY_MODE == "fetch":
        expected_titles = WIKI_PAGE_TITLES[:NUM_ARTICLES]

    missing_titles: List[str] = []
    if expected_titles is not None:
        missing_titles = [title for title in expected_titles if title not in cached_map]

    should_fetch = QUERY_MODE == "search" or bool(missing_titles)

    fetched_titles: List[str] = []
    if should_fetch:
        if QUERY_MODE == "search":
            base_titles = wikipedia.search(DOMAIN_QUERY, results=NUM_ARTICLES)
            print(f"-> fetching {len(base_titles)} Wikipedia articles for query: '{DOMAIN_QUERY}'")
        else:
            base_titles = missing_titles
            print(f"-> fetching {len(base_titles)} missing Wikipedia articles defined in configuration")

        next_index = _next_file_index(RAW_WIKI_DIR)
        for raw_title in base_titles:
            page = wikipedia.page(raw_title, auto_suggest=False)
            summary = page.summary
            fetched_titles.append(page.title)
            cached_map[page.title] = summary
            cached_entries.append((page.title, summary))

            safe_title = _sanitize_filename(page.title, f"article_{next_index}")
            filename = f"{next_index:02d}_{safe_title}.txt"
            file_content = (
                "Source: Wikipedia\n"
                f"Title: {page.title}\n"
                f"URL: {page.url}\n"
                f"Query: {DOMAIN_QUERY if QUERY_MODE == 'search' else 'fetch mode'}\n"
                "---\n"
                f"{summary}\n"
            )

            _save_document(RAW_WIKI_DIR, filename, file_content)
            print(f"    -> saved '{page.title}'")
            next_index += 1
    else:
        print("-> all requested Wikipedia articles already cached; reusing local files")

    if expected_titles is not None:
        titles = [title for title in expected_titles if title in cached_map]
    else:
        titles = fetched_titles if fetched_titles else [title for title, _ in cached_entries][:NUM_ARTICLES]

    summaries = [cached_map[title] for title in titles]

    print(f"    -> returning {len(summaries)} Wikipedia summaries.")
    return summaries, titles


def fetch_arxiv_data(ids: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """
    fetch a list of arXiv paper abstracts based on a query.
    
    this function can either search for the most relevant articles or 
    just fetch some predetermined ones. To define its behavior, go into
    config.py and read the section regarding to "entities and domain 
    configuration".
    
    Parameters
    ----------
    ids : Optional[list[str]]
        Optional explicit list of arXiv identifiers to fetch. When provided,
        this list will be used instead of the configured `ARXIV_PAPER_IDS`.

    Returns
    -------
    tuple[list[str], list[str]]
        Abstracts and their associated arXiv identifiers.
        
    """
    cached_entries = _load_cached_records(RAW_ARXIV_DIR, "ArXiv ID: ")
    cached_map = {paper_id: summary for paper_id, summary in cached_entries}

    expected_ids: Optional[List[str]] = None
    # honor an explicit list of ids provided by the caller
    if ids is not None:
        expected_ids = ids[:NUM_ARTICLES]
    elif QUERY_MODE == "fetch":
        expected_ids = ARXIV_PAPER_IDS[:NUM_ARTICLES]

    missing_ids: List[str] = []
    if expected_ids is not None:
        missing_ids = [paper_id for paper_id in expected_ids if paper_id not in cached_map]

    should_fetch = QUERY_MODE == "search" or bool(missing_ids)

    fetched_ids: List[str] = []
    if should_fetch:
        if QUERY_MODE == "search":
            print(f"-> fetching {NUM_ARTICLES} arXiv abstracts for query: '{DOMAIN_QUERY}'")
            search = arxiv.Search(
                query=DOMAIN_QUERY,
                max_results=NUM_ARTICLES,
                sort_by=arxiv.SortCriterion.Relevance,
            )
        else:
            print(f"-> fetching {len(missing_ids)} missing arXiv abstracts defined in configuration")
            search = arxiv.Search(id_list=missing_ids)

        arxiv_client = arxiv.Client()

        next_index = _next_file_index(RAW_ARXIV_DIR)
        for result in arxiv_client.results(search):
            summary = result.summary.replace("\n", " ")
            title = result.title.strip()
            arxiv_id = _extract_arxiv_id(result.entry_id)

            fetched_ids.append(arxiv_id)
            cached_map[arxiv_id] = summary
            cached_entries.append((arxiv_id, summary))

            safe_id = _sanitize_filename(arxiv_id, "arxiv")
            safe_title = _sanitize_filename(title, f"paper_{next_index}")
            filename = f"{next_index:02d}_{safe_id}_{safe_title}.txt"
            file_content = (
                "Source: arXiv\n"
                f"ArXiv ID: {arxiv_id}\n"
                f"Title: {title}\n"
                f"Query: {DOMAIN_QUERY if QUERY_MODE == 'search' else 'fetch mode'}\n"
                "---\n"
                f"{summary}\n"
            )

            _save_document(RAW_ARXIV_DIR, filename, file_content)
            print(f"    -> saved '{title}' ({arxiv_id})")
            next_index += 1
    else:
        print("-> all requested arXiv abstracts already cached; reusing local files")

    if expected_ids is not None:
        arxiv_ids = [paper_id for paper_id in expected_ids if paper_id in cached_map]
    else:
        arxiv_ids = fetched_ids if fetched_ids else [paper_id for paper_id, _ in cached_entries][:NUM_ARTICLES]

    abstracts = [cached_map[paper_id] for paper_id in arxiv_ids]

    print(f"    -> returning {len(abstracts)} arXiv abstracts.")
    return abstracts, arxiv_ids



if __name__ == "__main__":
    # test the file directly by running python3 -m src.kg_construction.fetch_data
    
    print("--- Testing Wikipedia Fetcher ---")
    raw_wiki, raw_wiki_titles = fetch_wiki_data()
    raw_arxiv, raw_arxiv_ids = fetch_arxiv_data()