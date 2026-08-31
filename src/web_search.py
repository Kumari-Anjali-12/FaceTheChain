"""
web_search.py
-------------
Stage 2 of the pipeline: given an image, run a genuine reverse-image
search against the web and return candidate matching pages/posts and
direct image URLs.

Uses Google Cloud Vision API's Web Detection feature, which crawls the
web for pages containing full or partial matches of the image and
returns their URLs, page titles, and best-guess labels.

Also includes `download_candidate_image()` for securely fetching candidate
image assets with strict SSRF protection, size limits, redirect validation,
and image format verification.
"""

from __future__ import annotations

import io
import ipaddress
import os
import socket
import urllib.parse
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image


@dataclass
class WebMatch:
    url: str = ""
    page_title: str | None = None
    match_type: str = ""  # "full_matching_image" | "partial_matching_image" | "visually_similar_image" | "pages_with_matching_images" | etc.
    score: float | None = None
    direct_image_url: str | None = None
    page_url: str | None = None

    def __post_init__(self) -> None:
        if not self.page_url and not self.direct_image_url and self.url:
            url_lower = self.url.lower()
            if any(url_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif")):
                self.direct_image_url = self.url
            else:
                self.page_url = self.url
        elif not self.url:
            self.url = self.page_url or self.direct_image_url or ""


def is_safe_url(url: str) -> bool:
    """
    Validates that a URL uses HTTP/HTTPS scheme and its hostname resolves to a
    public, non-private, non-loopback, non-link-local IP address.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False

    if parsed.scheme.lower() not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    hostname_lower = hostname.lower().strip()
    if hostname_lower in ("localhost", "localhost.localdomain") or hostname_lower.endswith(".localhost"):
        return False

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)

    try:
        addr_info = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except Exception:
        return False

    if not addr_info:
        return False

    for family, socktype, proto, canonname, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False

        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False

        if str(ip) in ("169.254.169.254", "0.0.0.0"):
            return False

    return True


def download_candidate_image(
    url: str,
    max_bytes: int = 10_000_000,
    timeout: int = 10,
) -> bytes | None:
    """
    Safely downloads a candidate image with strict SSRF protection, size limits,
    redirect validation, and decodable image content verification.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }

    current_url = url
    max_redirects = 3

    session = requests.Session()

    for redirect_count in range(max_redirects + 1):
        if not is_safe_url(current_url):
            return None

        try:
            response = session.get(
                current_url,
                headers=headers,
                stream=True,
                allow_redirects=False,
                timeout=timeout,
                verify=True,
            )
        except Exception:
            return None

        if response.status_code in (301, 302, 303, 307, 308):
            if redirect_count >= max_redirects:
                response.close()
                return None

            location = response.headers.get("Location")
            response.close()

            if not location:
                return None

            current_url = urllib.parse.urljoin(current_url, location)
            continue

        if response.status_code != 200:
            response.close()
            return None

        content_type = response.headers.get("Content-Type", "").lower().strip()
        if not content_type.startswith("image/"):
            response.close()
            return None

        content_length_hdr = response.headers.get("Content-Length")
        if content_length_hdr:
            try:
                if int(content_length_hdr) > max_bytes:
                    response.close()
                    return None
            except ValueError:
                pass

        downloaded = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    downloaded.extend(chunk)
                    if len(downloaded) > max_bytes:
                        response.close()
                        return None
        except Exception:
            response.close()
            return None
        finally:
            response.close()

        image_bytes = bytes(downloaded)
        if not image_bytes:
            return None

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()
            return image_bytes
        except Exception:
            return None

    return None


def search_by_image(image_path: str, max_results: int = 10) -> list[WebMatch]:
    """
    Reverse-image search via Google Cloud Vision Web Detection.

    Returns a deduplicated list of WebMatch objects, including direct image URLs
    and originating page URLs.
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
    raw_matches: list[WebMatch] = []

    # 1. Page-level matching images & full/partial sub-images
    for page in getattr(web_detection, "pages_with_matching_images", []):
        page_url = page.url if hasattr(page, "url") else None
        page_title = page.page_title or None if hasattr(page, "page_title") else None
        page_score = page.score if hasattr(page, "score") else None

        has_direct_imgs = False
        for img in getattr(page, "full_matching_images", []):
            has_direct_imgs = True
            raw_matches.append(
                WebMatch(
                    direct_image_url=img.url,
                    page_url=page_url,
                    page_title=page_title,
                    match_type="page_full_matching_image",
                    score=getattr(img, "score", page_score),
                )
            )

        for img in getattr(page, "partial_matching_images", []):
            has_direct_imgs = True
            raw_matches.append(
                WebMatch(
                    direct_image_url=img.url,
                    page_url=page_url,
                    page_title=page_title,
                    match_type="page_partial_matching_image",
                    score=getattr(img, "score", page_score),
                )
            )

        if not has_direct_imgs and page_url:
            raw_matches.append(
                WebMatch(
                    direct_image_url=None,
                    page_url=page_url,
                    page_title=page_title,
                    match_type="pages_with_matching_images",
                    score=page_score,
                )
            )

    # 2. Full matching images
    for img in getattr(web_detection, "full_matching_images", []):
        raw_matches.append(
            WebMatch(
                direct_image_url=img.url,
                page_url=None,
                page_title=None,
                match_type="full_matching_image",
                score=getattr(img, "score", None),
            )
        )

    # 3. Partial matching images
    for img in getattr(web_detection, "partial_matching_images", []):
        raw_matches.append(
            WebMatch(
                direct_image_url=img.url,
                page_url=None,
                page_title=None,
                match_type="partial_matching_image",
                score=getattr(img, "score", None),
            )
        )

    # 4. Visually similar images
    for img in getattr(web_detection, "visually_similar_images", []):
        raw_matches.append(
            WebMatch(
                direct_image_url=img.url,
                page_url=None,
                page_title=None,
                match_type="visually_similar_image",
                score=getattr(img, "score", None),
            )
        )

    # Deduplicate while preserving deterministic order
    seen = set()
    deduped_matches: list[WebMatch] = []
    for match in raw_matches:
        key = (match.direct_image_url, match.page_url)
        if key not in seen:
            seen.add(key)
            deduped_matches.append(match)

    return deduped_matches[:max_results]


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
        print(f"[{m.match_type}] img={m.direct_image_url} page={m.page_url} (score={m.score})")

    best = pick_best_social_match(results)
    if best:
        print(f"\nBest social match: {best.url}")
