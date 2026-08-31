"""
tests/test_pipeline.py
-----------------------
Unit and integration tests for the Phase 3B end-to-end pipeline in src/pipeline.py.
"""

import hashlib
import io
import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.face_id import CandidateVerificationResult, FaceEncodingResult
from src.pipeline import (
    PipelineResult,
    build_canonical_evidence,
    canonicalize_and_hash,
    generate_deterministic_record_id,
    run_pipeline,
)
from src.web_search import WebMatch


def create_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (30, 30), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def dummy_face_result():
    return FaceEncodingResult(
        image_path="dummy_path.jpg",
        face_location=(10, 20, 30, 40),
        encoding=[0.0] * 128,
        image_sha256="1111111111111111111111111111111111111111111111111111111111111111",
    )


@pytest.fixture
def dummy_web_matches():
    return [
        WebMatch(
            url="https://example.com/page1",
            match_type="full_matching_image",
            score=0.95,
            page_title="Page 1 Title",
            direct_image_url="https://example.com/image1.jpg",
        ),
        WebMatch(
            url="https://example.com/page2",
            match_type="partial_matching_image",
            score=0.85,
            page_title="Page 2 Title",
            direct_image_url="https://example.com/image2.jpg",
        ),
    ]


# --- Test 1: test_end_to_end_verified_pipeline ---

@patch("src.pipeline.encode_face")
@patch("src.pipeline.search_by_image")
@patch("src.pipeline.download_candidate_image")
@patch("src.pipeline.verify_candidate_face")
def test_end_to_end_verified_pipeline(
    mock_verify_face,
    mock_download,
    mock_search,
    mock_encode,
    dummy_face_result,
    dummy_web_matches,
):
    os.environ["BLOCKCHAIN_MODE"] = "local"
    mock_encode.return_value = dummy_face_result
    mock_search.return_value = dummy_web_matches
    dummy_bytes = create_dummy_image_bytes()
    mock_download.return_value = dummy_bytes

    mock_verify_face.return_value = CandidateVerificationResult(
        is_match=True,
        best_distance=0.15,
        best_face_location=(5, 15, 25, 35),
        number_of_detected_faces=1,
    )

    res = run_pipeline("dummy_path.jpg", tolerance=0.6)

    assert isinstance(res, PipelineResult)
    assert res.status == "VERIFIED_MATCH_REGISTERED"
    assert res.input_image_sha256 == dummy_face_result.image_sha256
    assert res.candidates_evaluated == 1
    assert res.verified_candidate is not None
    assert res.verified_candidate["page_url"] == "https://example.com/page1"
    assert res.verified_candidate["image_url"] == "https://example.com/image1.jpg"
    assert res.canonical_evidence is not None
    assert res.canonical_evidence["schema_version"] == "1.0"
    assert res.evidence_hash is not None
    assert res.record_id is not None
    assert len(res.record_id) == 32
    assert res.on_chain_record is not None
    assert res.original_verified is True
    assert res.tamper_detection_successful is True


# --- Test 2: test_pipeline_no_web_matches ---

@patch("src.pipeline.encode_face")
@patch("src.pipeline.search_by_image")
def test_pipeline_no_web_matches(mock_search, mock_encode, dummy_face_result):
    mock_encode.return_value = dummy_face_result
    mock_search.return_value = []

    res = run_pipeline("dummy_path.jpg")

    assert res.status == "NO_CANDIDATE_FOUND"
    assert res.input_image_sha256 == dummy_face_result.image_sha256
    assert res.record_id is None
    assert res.verified_candidate is None


# --- Test 3: test_pipeline_unverified_faces_no_blockchain_upload ---

@patch("src.pipeline.encode_face")
@patch("src.pipeline.search_by_image")
@patch("src.pipeline.download_candidate_image")
@patch("src.pipeline.verify_candidate_face")
@patch("src.pipeline.upload_record")
def test_pipeline_unverified_faces_no_blockchain_upload(
    mock_upload,
    mock_verify_face,
    mock_download,
    mock_search,
    mock_encode,
    dummy_face_result,
    dummy_web_matches,
):
    mock_encode.return_value = dummy_face_result
    mock_search.return_value = dummy_web_matches
    mock_download.return_value = create_dummy_image_bytes()

    # Face detected, but distance (0.75) > tolerance (0.6)
    mock_verify_face.return_value = CandidateVerificationResult(
        is_match=False,
        best_distance=0.75,
        best_face_location=(0, 10, 10, 0),
        number_of_detected_faces=1,
    )

    res = run_pipeline("dummy_path.jpg", tolerance=0.6)

    assert res.status == "NO_VERIFIED_FACE_MATCH"
    assert res.candidates_evaluated == 2
    assert res.record_id is None
    # Crucial security check: upload_record MUST NEVER be called
    mock_upload.assert_not_called()


# --- Test 4: test_pipeline_ssrf_download_failure_fallback ---

@patch("src.pipeline.encode_face")
@patch("src.pipeline.search_by_image")
@patch("src.pipeline.download_candidate_image")
@patch("src.pipeline.verify_candidate_face")
def test_pipeline_ssrf_download_failure_fallback(
    mock_verify_face,
    mock_download,
    mock_search,
    mock_encode,
    dummy_web_matches,
):
    os.environ["BLOCKCHAIN_MODE"] = "local"
    unique_face_res = FaceEncodingResult(
        image_path="fallback_path.jpg",
        face_location=(10, 20, 30, 40),
        encoding=[0.0] * 128,
        image_sha256="2222222222222222222222222222222222222222222222222222222222222222",
    )
    mock_encode.return_value = unique_face_res
    mock_search.return_value = dummy_web_matches

    # Candidate 1 fails download (SSRF / network error); Candidate 2 succeeds
    img = Image.new("RGB", (30, 30), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    dummy_bytes = buf.getvalue()
    mock_download.side_effect = [None, dummy_bytes]

    mock_verify_face.return_value = CandidateVerificationResult(
        is_match=True,
        best_distance=0.2,
        best_face_location=(1, 2, 3, 4),
        number_of_detected_faces=1,
    )

    res = run_pipeline("fallback_path.jpg")

    assert res.status == "VERIFIED_MATCH_REGISTERED"
    assert res.candidates_evaluated == 2
    assert res.verified_candidate["page_url"] == "https://example.com/page2"
    assert res.verified_candidate["image_url"] == "https://example.com/image2.jpg"


# --- Test 5: test_canonical_json_hash_determinism ---

def test_canonical_json_hash_determinism():
    evidence1 = {
        "z_field": "last",
        "a_field": "first",
        "nested": {"beta": 2, "alpha": 1},
        "score": 0.95,
        "is_match": True,
    }
    evidence2 = {
        "a_field": "first",
        "score": 0.95,
        "nested": {"alpha": 1, "beta": 2},
        "is_match": True,
        "z_field": "last",
    }

    hash1, bytes1 = canonicalize_and_hash(evidence1)
    hash2, bytes2 = canonicalize_and_hash(evidence2)

    assert hash1 == hash2
    assert bytes1 == bytes2
    assert bytes1.decode("utf-8") == '{"a_field":"first","is_match":true,"nested":{"alpha":1,"beta":2},"score":0.95,"z_field":"last"}'


# --- Test 6: test_deterministic_record_id ---

def test_deterministic_record_id():
    input_sha = "a" * 64
    candidate_sha = "b" * 64

    rec_id1 = generate_deterministic_record_id(input_sha, candidate_sha)
    rec_id2 = generate_deterministic_record_id(input_sha, candidate_sha)

    assert rec_id1 == rec_id2
    assert len(rec_id1) == 32
    assert isinstance(rec_id1, str)

    expected_src = f"faceid:{input_sha}:{candidate_sha}"
    expected_hex = hashlib.sha256(expected_src.encode("utf-8")).hexdigest()[:32]
    assert rec_id1 == expected_hex


# --- Test 7: test_tamper_detection_assertion ---

@patch("src.pipeline.upload_record")
@patch("src.pipeline.verify_record")
def test_tamper_detection_assertion(mock_verify, mock_upload):
    original_evidence = {
        "schema_version": "1.0",
        "input_image_sha256": "1111",
        "candidate_image_sha256": "2222",
        "face_distance": 0.1234,
        "is_face_match": True,
        "timestamp": "2026-08-31T21:00:00Z",
    }

    orig_hash, _ = canonicalize_and_hash(original_evidence)
    record_id = generate_deterministic_record_id("1111", "2222")

    # Simulate verify_record output: returns True for orig_hash, False for any tampered hash
    def verify_side_effect(rec_id, check_hash):
        return check_hash == orig_hash

    mock_verify.side_effect = verify_side_effect

    # 1. Original evidence verification succeeds
    assert mock_verify(record_id, orig_hash) is True

    # 2. Tampered evidence (altering face distance) fails verification
    tampered_evidence = dict(original_evidence)
    tampered_evidence["face_distance"] = 0.0500
    tampered_hash, _ = canonicalize_and_hash(tampered_evidence)

    assert tampered_hash != orig_hash
    assert mock_verify(record_id, tampered_hash) is False
