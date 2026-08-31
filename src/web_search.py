"""
web_search.py
-------------
Stage 2 of the pipeline: given an image, run a genuine reverse-image
search against the web and return candidate matching pages/posts.

Uses Google Cloud Vision API's Web Detection feature, which crawls the
web for pages containing full or partial matches of the image and
returns their URLs, page titles, and best-guess labels. This is a real
search call, not a lookup table.

Requires a Google Cloud project with the Vision API enabled and a
service-account key. Set GOOGLE_APPLICATION_CREDENTIALS to the path
of your key JSON, or export GOOGLE_VISION_API_KEY for the simpler
REST API-key flow (see README).

IMPORTANT (ethics/consent):
Run this only against images you have the right to search — e.g. your
own photo, or a teammate's with their explicit consent for the demo.
Do not use this to locate a stranger's social media accounts without
their permission.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class WebMatch:
    url: str
    page_title: str | None
    match_type: str  # "full_matching_image" | "partial_matching_image" | "pages_with_matching_images"
    score: float | None = None


def search_by_image(image_path: str, max_results: int = 10) -> list[WebMatch]:
    """
    Reverse-image search via Google Cloud Vision Web Detection.

    Returns a list of WebMatch objects, best matches first.
    """
    try:
        from google.cloud import vision
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-vision not installed. Run: pip install google-cloud-vision"
        ) from e

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. Point it at your "
            "GCP service-account key JSON before calling search_by_image()."
        )

    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    response = client.web_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    web_detection = response.web_detection
    matches: list[WebMatch] = []

    # Full matches are the strongest signal (near-identical image found elsewhere).
    for page in web_detection.pages_with_matching_images[:max_results]:
        matches.append(
            WebMatch(
                url=page.url,
                page_title=page.page_title or None,
                match_type="pages_with_matching_images",
                score=page.score if hasattr(page, "score") else None,
            )
        )

    for img in web_detection.full_matching_images[:max_results]:
        matches.append(
            WebMatch(url=img.url, page_title=None, match_type="full_matching_image", score=img.score)
        )

    for img in web_detection.partial_matching_images[:max_results]:
        matches.append(
            WebMatch(url=img.url, page_title=None, match_type="partial_matching_image", score=img.score)
        )

    return matches[:max_results]


def pick_best_social_match(matches: list[WebMatch]) -> WebMatch | None:
    """
    Filter candidate matches down to ones that look like social media
    posts/profiles, and return the highest-confidence one.
    """
    social_domains = (
        "instagram.com", "twitter.com", "x.com", "facebook.com",
        "linkedin.com", "tiktok.com", "reddit.com", "pinterest.com",
        "threads.net",
    )
    social_matches = [m for m in matches if any(d in m.url for d in social_domains)]
    pool = social_matches or matches
    if not pool:
        return None
    return sorted(pool, key=lambda m: (m.score or 0), reverse=True)[0]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python web_search.py <image_path>")
        sys.exit(1)

    results = search_by_image(sys.argv[1])
    for m in results:
        print(f"[{m.match_type}] {m.url} (score={m.score})")

    best = pick_best_social_match(results)
    if best:
        print(f"\nBest social match: {best.url}")
