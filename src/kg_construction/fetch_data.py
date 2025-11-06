import re
from pathlib import Path

import arxiv
import wikipedia

from src.config import (
    DOMAIN_QUERY,
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


def fetch_wiki_data(query, num_articles=5):
    """
    fetch a list of Wikipedia page summaries based on a query.
    
    this function can either search for the most relevant articles or 
    just fetch some predetermined ones. To define its behavior, go into
    config.py and read the section regarding to "entities and domain 
    configuration".
    """
    
    print(f"-> fetching {num_articles} Wikipedia articles for query: '{query}'")
    
    # search for page titles
    if QUERY_MODE == "fetch":
        page_titles = WIKI_PAGE_TITLES[:num_articles]
    else:
        page_titles = wikipedia.search(query, results=num_articles)
    
    # get the summary for each page
    summaries = []
    saved_count = 0
    for title in page_titles:
        page = wikipedia.page(title, auto_suggest=False)

        summary = page.summary
        summaries.append(summary)
        saved_count += 1

        safe_title = _sanitize_filename(page.title, f"article_{saved_count}")
        filename = f"{saved_count:02d}_{safe_title}.txt"
        file_content = (
            "Source: Wikipedia\n"
            f"Title: {page.title}\n"
            f"URL: {page.url}\n"
            f"Query: {query}\n"
            "---\n"
            f"{summary}\n"
        )

        saved_path = _save_document(RAW_WIKI_DIR, filename, file_content)
        print(f"    -> saved '{page.title}'")
        
    print(f"\n    -> found {len(summaries)} Wikipedia summaries.")
    return summaries


def fetch_arxiv_data(query, num_articles=5):
    """
    fetch a list of arXiv paper abstracts based on a query.
    
    this function can either search for the most relevant articles or 
    just fetch some predetermined ones. To define its behavior, go into
    config.py and read the section regarding to "entities and domain 
    configuration".
    """
    
    print(f"-> fetching {num_articles} arXiv abstracts for query: '{query}'")
    
    # search for papers
    if QUERY_MODE == "fetch":
        search = arxiv.Search(id_list=ARXIV_PAPER_IDS[:num_articles])
    else:
        search = arxiv.Search(
            query=query,
            max_results=num_articles,
            sort_by=arxiv.SortCriterion.Relevance
        )
    
    arxiv_client = arxiv.Client()
    
    # get the abstracts for each paper
    abstracts = []
    for index, result in enumerate(arxiv_client.results(search), start=1):
        summary = result.summary.replace("\n", " ")  # Clean newlines for downstream use
        title = result.title.strip()
        arxiv_id = _extract_arxiv_id(result.entry_id)

        abstracts.append(summary)

        safe_id = _sanitize_filename(arxiv_id, "arxiv")
        safe_title = _sanitize_filename(title, f"paper_{index}")
        filename = f"{index:02d}_{safe_id}_{safe_title}.txt"
        file_content = (
            "Source: arXiv\n"
            f"ArXiv ID: {arxiv_id}\n"
            f"Title: {title}\n"
            f"Query: {query}\n"
            "---\n"
            f"{summary}\n"
        )

        saved_path = _save_document(RAW_ARXIV_DIR, filename, file_content)
        print(f"    -> saved '{title}' ({arxiv_id})")
        
    print(f"\n    -> found {len(abstracts)} arXiv abstracts.")
    return abstracts



if __name__ == "__main__":
    # test the file directly by running python3 -m src.kg_construction.fetch_data
    
    print("--- Testing Wikipedia Fetcher ---")
    rawwiki = fetch_wiki_data(DOMAIN_QUERY, num_articles=2)
    raw_arxiv = fetch_arxiv_data(DOMAIN_QUERY, num_articles=2)