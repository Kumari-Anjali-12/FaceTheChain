"""
tests/test_candidate_verification.py
--------------------------------------
Unit tests for candidate face verification in src/face_id.py.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.face_id import (
    CandidateVerificationResult,
    _load_image_from_input,
    verify_candidate_face,
)


def create_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (30, 30), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def input_encoding():
    return np.zeros(128, dtype=np.float64)


# --- Unit Tests for verify_candidate_face ---

@patch("face_recognition.face_encodings")
@patch("face_recognition.face_locations")
def test_matching_face(mock_locations, mock_encodings, input_encoding):
    mock_locations.return_value = [(0, 10, 10, 0)]
    mock_encodings.return_value = [np.zeros(128, dtype=np.float64)]

    dummy_bytes = create_dummy_image_bytes()
    result = verify_candidate_face(input_encoding, dummy_bytes, tolerance=0.6)

    assert isinstance(result, CandidateVerificationResult)
    assert result.is_match is True
    assert result.best_distance == pytest.approx(0.0)
    assert result.best_face_location == (0, 10, 10, 0)
    assert result.number_of_detected_faces == 1


@patch("face_recognition.face_encodings")
@patch("face_recognition.face_locations")
def test_different_face(mock_locations, mock_encodings, input_encoding):
    mock_locations.return_value = [(0, 10, 10, 0)]
    # Distance will be sqrt(128) ~= 11.31 > 0.6
    mock_encodings.return_value = [np.ones(128, dtype=np.float64)]

    dummy_bytes = create_dummy_image_bytes()
    result = verify_candidate_face(input_encoding, dummy_bytes, tolerance=0.6)

    assert result.is_match is False
    assert result.best_distance == pytest.approx(np.sqrt(128))
    assert result.best_face_location == (0, 10, 10, 0)
    assert result.number_of_detected_faces == 1


@patch("face_recognition.face_encodings")
@patch("face_recognition.face_locations")
def test_multiple_faces_one_match(mock_locations, mock_encodings, input_encoding):
    # 3 faces detected in candidate image
    mock_locations.return_value = [(0, 10, 10, 0), (20, 30, 30, 20), (40, 50, 50, 40)]

    enc1 = np.ones(128, dtype=np.float64)  # dist ~= 11.31
    enc2 = np.full(128, 0.02, dtype=np.float64)  # dist ~= 0.226 (MATCH!)
    enc3 = np.ones(128, dtype=np.float64) * 2  # dist ~= 22.62

    mock_encodings.return_value = [enc1, enc2, enc3]

    dummy_bytes = create_dummy_image_bytes()
    result = verify_candidate_face(input_encoding, dummy_bytes, tolerance=0.6)

    assert result.is_match is True
    assert result.best_distance == pytest.approx(np.linalg.norm(input_encoding - enc2))
    assert result.best_face_location == (20, 30, 30, 20)
    assert result.number_of_detected_faces == 3


@patch("face_recognition.face_encodings")
@patch("face_recognition.face_locations")
def test_multiple_faces_none_match(mock_locations, mock_encodings, input_encoding):
    mock_locations.return_value = [(0, 10, 10, 0), (20, 30, 30, 20)]

    enc1 = np.full(128, 0.7, dtype=np.float64)  # dist ~= 7.92 > 0.6
    enc2 = np.ones(128, dtype=np.float64)  # dist ~= 11.31 > 0.6

    mock_encodings.return_value = [enc1, enc2]

    dummy_bytes = create_dummy_image_bytes()
    result = verify_candidate_face(input_encoding, dummy_bytes, tolerance=0.6)

    assert result.is_match is False
    assert result.best_distance == pytest.approx(np.linalg.norm(input_encoding - enc1))
    assert result.best_face_location == (0, 10, 10, 0)
    assert result.number_of_detected_faces == 2


@patch("face_recognition.face_locations")
def test_zero_face_image(mock_locations, input_encoding):
    mock_locations.return_value = []

    dummy_bytes = create_dummy_image_bytes()
    result = verify_candidate_face(input_encoding, dummy_bytes, tolerance=0.6)

    assert result.is_match is False
    assert result.best_distance is None
    assert result.best_face_location is None
    assert result.number_of_detected_faces == 0


def test_invalid_image_bytes(input_encoding):
    corrupt_bytes = b"CORRUPTED_IMAGE_BYTES_12345"
    result = verify_candidate_face(input_encoding, corrupt_bytes, tolerance=0.6)

    assert result.is_match is False
    assert result.best_distance is None
    assert result.best_face_location is None
    assert result.number_of_detected_faces == 0


def test_nonexistent_file_path(input_encoding):
    result = verify_candidate_face(input_encoding, "nonexistent_file_xyz_123.jpg", tolerance=0.6)

    assert result.is_match is False
    assert result.best_distance is None
    assert result.best_face_location is None
    assert result.number_of_detected_faces == 0


@patch("face_recognition.face_encodings")
@patch("face_recognition.face_locations")
def test_tolerance_behavior(mock_locations, mock_encodings, input_encoding):
    mock_locations.return_value = [(0, 10, 10, 0)]
    enc = np.full(128, 0.04, dtype=np.float64)  # dist = sqrt(128 * 0.0016) ~= 0.4525
    mock_encodings.return_value = [enc]

    dummy_bytes = create_dummy_image_bytes()

    # Pass with loose tolerance (0.6)
    res_loose = verify_candidate_face(input_encoding, dummy_bytes, tolerance=0.6)
    assert res_loose.is_match is True
    assert res_loose.best_distance == pytest.approx(np.linalg.norm(input_encoding - enc))

    # Fail with strict tolerance (0.3)
    res_strict = verify_candidate_face(input_encoding, dummy_bytes, tolerance=0.3)
    assert res_strict.is_match is False
    assert res_strict.best_distance == pytest.approx(np.linalg.norm(input_encoding - enc))


def test_malformed_input(input_encoding):
    assert verify_candidate_face(input_encoding, None).is_match is False
    assert verify_candidate_face(input_encoding, 12345).is_match is False
    assert verify_candidate_face(input_encoding, []).is_match is False


def test_list_input_encoding():
    list_encoding = [0.0] * 128
    dummy_bytes = create_dummy_image_bytes()

    with patch("face_recognition.face_locations", return_value=[(0, 10, 10, 0)]):
        with patch("face_recognition.face_encodings", return_value=[np.zeros(128)]):
            res = verify_candidate_face(list_encoding, dummy_bytes)
            assert res.is_match is True
            assert res.best_distance == pytest.approx(0.0)


def test_candidate_verification_result_repr():
    res = CandidateVerificationResult(
        is_match=True,
        best_distance=0.123456,
        best_face_location=(10, 20, 30, 40),
        number_of_detected_faces=2,
    )
    repr_str = repr(res)
    assert "is_match=True" in repr_str
    assert "best_distance=0.1235" in repr_str
    assert "0.123456" not in repr_str  # rounded representation
    assert "array" not in repr_str  # raw embedding not exposed
