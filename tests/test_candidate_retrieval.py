"""
tests/test_candidate_retrieval.py
----------------------------------
Unit tests for candidate image retrieval and SSRF-hardened image downloader in web_search.py.
"""

import io
import ipaddress
import socket
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.web_search import (
    WebMatch,
    download_candidate_image,
    is_ip_safe,
    is_safe_url,
    search_by_image,
)


def create_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (20, 20), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Mock Classes for Google Cloud Vision API ---

class MockWebImage:
    def __init__(self, url: str, score: float = 0.9):
        self.url = url
        self.score = score


class MockWebPage:
    def __init__(
        self,
        url: str,
        page_title: str,
        full_matching_images=None,
        partial_matching_images=None,
        score: float = 0.95,
    ):
        self.url = url
        self.page_title = page_title
        self.full_matching_images = full_matching_images or []
        self.partial_matching_images = partial_matching_images or []
        self.score = score


class MockWebDetection:
    def __init__(
        self,
        pages_with_matching_images=None,
        full_matching_images=None,
        partial_matching_images=None,
        visually_similar_images=None,
    ):
        self.pages_with_matching_images = pages_with_matching_images or []
        self.full_matching_images = full_matching_images or []
        self.partial_matching_images = partial_matching_images or []
        self.visually_similar_images = visually_similar_images or []


# --- Tests for WebMatch & Vision Extraction ---

def test_webmatch_url_property_compatibility():
    match1 = WebMatch(page_url="https://example.com/page")
    assert match1.url == "https://example.com/page"

    match2 = WebMatch(direct_image_url="https://example.com/img.jpg")
    assert match2.url == "https://example.com/img.jpg"

    match3 = WebMatch(url="https://example.com/legacy.png")
    assert match3.direct_image_url == "https://example.com/legacy.png"
    assert match3.url == "https://example.com/legacy.png"


from unittest.mock import mock_open

@patch("os.environ.get", return_value="/fake/path.json")
@patch("google.cloud.vision.ImageAnnotatorClient")
def test_extract_full_matching_images(mock_client_cls, mock_env):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.error.message = ""
    mock_response.web_detection = MockWebDetection(
        full_matching_images=[MockWebImage("https://example.com/full.jpg")]
    )
    mock_client.web_detection.return_value = mock_response

    with patch("builtins.open", mock_open(read_data=b"fake_image_bytes")):
        matches = search_by_image("dummy_path.jpg")

    assert len(matches) == 1
    assert matches[0].match_type == "full_matching_image"
    assert matches[0].direct_image_url == "https://example.com/full.jpg"


@patch("os.environ.get", return_value="/fake/path.json")
@patch("google.cloud.vision.ImageAnnotatorClient")
def test_extract_partial_matching_images(mock_client_cls, mock_env):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.error.message = ""
    mock_response.web_detection = MockWebDetection(
        partial_matching_images=[MockWebImage("https://example.com/partial.jpg")]
    )
    mock_client.web_detection.return_value = mock_response

    with patch("builtins.open", mock_open(read_data=b"fake_image_bytes")):
        matches = search_by_image("dummy_path.jpg")

    assert len(matches) == 1
    assert matches[0].match_type == "partial_matching_image"
    assert matches[0].direct_image_url == "https://example.com/partial.jpg"


@patch("os.environ.get", return_value="/fake/path.json")
@patch("google.cloud.vision.ImageAnnotatorClient")
def test_extract_visually_similar_images(mock_client_cls, mock_env):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.error.message = ""
    mock_response.web_detection = MockWebDetection(
        visually_similar_images=[MockWebImage("https://example.com/similar.jpg")]
    )
    mock_client.web_detection.return_value = mock_response

    with patch("builtins.open", mock_open(read_data=b"fake_image_bytes")):
        matches = search_by_image("dummy_path.jpg")

    assert len(matches) == 1
    assert matches[0].match_type == "visually_similar_image"
    assert matches[0].direct_image_url == "https://example.com/similar.jpg"


@patch("os.environ.get", return_value="/fake/path.json")
@patch("google.cloud.vision.ImageAnnotatorClient")
def test_extract_page_level_image_urls(mock_client_cls, mock_env):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.error.message = ""
    mock_page = MockWebPage(
        url="https://example.com/article",
        page_title="Example Article",
        full_matching_images=[MockWebImage("https://example.com/article_img.jpg")],
    )
    mock_response.web_detection = MockWebDetection(pages_with_matching_images=[mock_page])
    mock_client.web_detection.return_value = mock_response

    with patch("builtins.open", mock_open(read_data=b"fake_image_bytes")):
        matches = search_by_image("dummy_path.jpg")

    assert len(matches) == 1
    assert matches[0].match_type == "page_full_matching_image"
    assert matches[0].direct_image_url == "https://example.com/article_img.jpg"
    assert matches[0].page_url == "https://example.com/article"
    assert matches[0].page_title == "Example Article"


@patch("os.environ.get", return_value="/fake/path.json")
@patch("google.cloud.vision.ImageAnnotatorClient")
def test_deduplication(mock_client_cls, mock_env):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.error.message = ""
    mock_response.web_detection = MockWebDetection(
        full_matching_images=[
            MockWebImage("https://example.com/same.jpg"),
            MockWebImage("https://example.com/same.jpg"),
        ],
        partial_matching_images=[
            MockWebImage("https://example.com/same.jpg"),
        ],
    )
    mock_client.web_detection.return_value = mock_response

    with patch("builtins.open", mock_open(read_data=b"fake_image_bytes")):
        matches = search_by_image("dummy_path.jpg")

    assert len(matches) == 1
    assert matches[0].direct_image_url == "https://example.com/same.jpg"


# --- SSRF & Security Tests ---

@patch("socket.getaddrinfo")
def test_is_safe_url_public_ip(mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]
    assert is_safe_url("https://example.com/photo.jpg") is True


@patch("socket.getaddrinfo")
def test_is_safe_url_rejections(mock_getaddrinfo):
    # Non-HTTP schemes
    assert is_safe_url("file:///etc/passwd") is False
    assert is_safe_url("ftp://example.com/img.png") is False
    assert is_safe_url("gopher://example.com/") is False

    # Localhost string check
    assert is_safe_url("http://localhost/test.jpg") is False
    assert is_safe_url("http://localhost.localdomain/test.jpg") is False
    assert is_safe_url("http://sub.localhost/test.jpg") is False

    # Loopback IP (127.0.0.1)
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
    ]
    assert is_safe_url("http://127.0.0.1/test.jpg") is False

    # Cloud metadata endpoint (169.254.169.254)
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))
    ]
    assert is_safe_url("http://169.254.169.254/latest/meta-data/") is False

    # Private IP ranges (10.0.0.1, 192.168.1.1, 172.16.0.1)
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 80))
    ]
    assert is_safe_url("http://10.0.0.1/private.png") is False

    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 80))
    ]
    assert is_safe_url("http://192.168.1.1/private.png") is False

    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("172.16.0.1", 80))
    ]
    assert is_safe_url("http://172.16.0.1/private.png") is False


# --- Downloader Tests ---

@patch("socket.getaddrinfo")
@patch("requests.Session.get")
def test_successful_image_download(mock_get, mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]
    valid_bytes = create_dummy_image_bytes()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "image/png", "Content-Length": str(len(valid_bytes))}
    mock_response.iter_content.return_value = [valid_bytes]
    mock_get.return_value = mock_response

    downloaded = download_candidate_image("https://example.com/test.png")
    assert downloaded == valid_bytes


@patch("socket.getaddrinfo")
@patch("requests.Session.get")
def test_content_size_limit_rejection(mock_get, mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    # Content-Length header exceeding limit
    mock_response.headers = {"Content-Type": "image/png", "Content-Length": "20000000"}
    mock_get.return_value = mock_response

    result = download_candidate_image("https://example.com/large.png", max_bytes=10_000_000)
    assert result is None

    # Exceeding during stream
    mock_response.headers = {"Content-Type": "image/png"}
    mock_response.iter_content.return_value = [b"A" * 6000, b"B" * 6000]
    result_stream = download_candidate_image("https://example.com/large.png", max_bytes=10_000)
    assert result_stream is None


@patch("socket.getaddrinfo")
@patch("requests.Session.get")
def test_non_image_mime_rejection(mock_get, mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_get.return_value = mock_response

    result = download_candidate_image("https://example.com/page.html")
    assert result is None


@patch("socket.getaddrinfo")
@patch("requests.Session.get")
def test_invalid_image_bytes_rejection(mock_get, mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "image/jpeg"}
    mock_response.iter_content.return_value = [b"THIS_IS_NOT_AN_IMAGE_PAYLOAD"]
    mock_get.return_value = mock_response

    result = download_candidate_image("https://example.com/corrupt.jpg")
    assert result is None


@patch("socket.getaddrinfo")
@patch("requests.Session.get")
def test_graceful_http_error(mock_get, mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]

    # HTTP 404
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    result = download_candidate_image("https://example.com/missing.png")
    assert result is None


@patch("socket.getaddrinfo")
@patch("requests.Session.get")
def test_redirect_limit(mock_get, mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]

    # 4 consecutive redirects
    mock_resp_redirect = MagicMock()
    mock_resp_redirect.status_code = 302
    mock_resp_redirect.headers = {"Location": "https://example.com/next"}
    mock_get.return_value = mock_resp_redirect

    result = download_candidate_image("https://example.com/start")
    assert result is None


@patch("socket.getaddrinfo")
@patch("requests.Session.get")
def test_redirect_target_ssrf_validation(mock_get, mock_getaddrinfo):
    def getaddrinfo_side_effect(host, port, *args, **kwargs):
        if host == "example.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        elif host in ("127.0.0.1", "localhost", "internal.bank.com"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 80))]

    mock_getaddrinfo.side_effect = getaddrinfo_side_effect

    mock_resp_redirect = MagicMock()
    mock_resp_redirect.status_code = 302
    mock_resp_redirect.headers = {"Location": "http://127.0.0.1/internal.png"}
    mock_get.return_value = mock_resp_redirect

    # First URL example.com resolves to public IP, but redirect location goes to 127.0.0.1
    result = download_candidate_image("https://example.com/initial.png")
    assert result is None


def test_is_ip_safe_comprehensive():
    # Safe public IPs
    assert is_ip_safe(ipaddress.ip_address("93.184.216.34")) is True
    assert is_ip_safe(ipaddress.ip_address("8.8.8.8")) is True
    assert is_ip_safe(ipaddress.ip_address("2001:4860:4860::8888")) is True

    # Loopback
    assert is_ip_safe(ipaddress.ip_address("127.0.0.1")) is False
    assert is_ip_safe(ipaddress.ip_address("::1")) is False

    # Private IPv4
    assert is_ip_safe(ipaddress.ip_address("10.0.0.1")) is False
    assert is_ip_safe(ipaddress.ip_address("172.16.0.1")) is False
    assert is_ip_safe(ipaddress.ip_address("192.168.1.1")) is False
    assert is_ip_safe(ipaddress.ip_address("100.64.0.1")) is False

    # Link-local & Metadata
    assert is_ip_safe(ipaddress.ip_address("169.254.169.254")) is False
    assert is_ip_safe(ipaddress.ip_address("fe80::1")) is False

    # IPv6 ULA & Unspecified
    assert is_ip_safe(ipaddress.ip_address("fc00::1")) is False
    assert is_ip_safe(ipaddress.ip_address("::")) is False
    assert is_ip_safe(ipaddress.ip_address("0.0.0.0")) is False

    # IPv4-mapped IPv6
    assert is_ip_safe(ipaddress.ip_address("::ffff:127.0.0.1")) is False
    assert is_ip_safe(ipaddress.ip_address("::ffff:10.0.0.1")) is False
    assert is_ip_safe(ipaddress.ip_address("::ffff:169.254.169.254")) is False
    assert is_ip_safe(ipaddress.ip_address("::ffff:93.184.216.34")) is True


def test_is_safe_url_credentials_and_ports():
    # User credentials in URL
    assert is_safe_url("http://user:pass@example.com/image.jpg") is False
    assert is_safe_url("http://user@example.com/image.jpg") is False

    # Invalid ports
    assert is_safe_url("http://example.com:0/image.jpg") is False
    assert is_safe_url("http://example.com:65536/image.jpg") is False


@patch("socket.getaddrinfo")
@patch("requests.Session.get")
def test_https_to_http_redirect_downgrade_rejected(mock_get, mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]

    mock_resp_redirect = MagicMock()
    mock_resp_redirect.status_code = 302
    mock_resp_redirect.headers = {"Location": "http://example.com/insecure_target.png"}
    mock_get.return_value = mock_resp_redirect

    # HTTPS URL redirecting to HTTP URL should be disallowed
    result = download_candidate_image("https://example.com/secure_start.png")
    assert result is None


@patch("socket.getaddrinfo")
@patch("requests.Session.get")
def test_decompression_bomb_rejection(mock_get, mock_getaddrinfo):
    mock_getaddrinfo.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
    ]

    # Create dummy image larger than 25 Megapixels (6000 x 5000 = 30 MP)
    large_img = Image.new("RGB", (6000, 5000), color="red")
    buf = io.BytesIO()
    large_img.save(buf, format="PNG")
    bomb_bytes = buf.getvalue()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "image/png"}
    mock_response.iter_content.return_value = [bomb_bytes]
    mock_get.return_value = mock_response

    # Exceeding 25 MP pixel limit should return None
    result = download_candidate_image("https://example.com/bomb.png")
    assert result is None
