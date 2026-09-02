"""
face_id.py
----------
Stage 1 of the pipeline: detect a face in an input image and produce
a numeric encoding (embedding) that can be used for downstream matching.

Uses the `face_recognition` library (built on dlib's ResNet face
recognition model), which is the standard, well-documented choice for
a hackathon-scale project. Swap in DeepFace / InsightFace if you want
higher accuracy or GPU acceleration.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import face_recognition
import numpy as np
from PIL import Image, ImageOps

MAX_IMAGE_PIXELS = 25_000_000  # 25 Megapixels limit to prevent decompression bombs


@dataclass
class FaceEncodingResult:
    image_path: str
    encoding: np.ndarray
    face_location: tuple  # (top, right, bottom, left)
    image_sha256: str = field(default="")

    def __post_init__(self):
        if not self.image_sha256:
            self.image_sha256 = _sha256_of_file(self.image_path)

    def encoding_as_list(self) -> list:
        """JSON-serializable form of the 128-d embedding."""
        return self.encoding.tolist()


@dataclass
class CandidateVerificationResult:
    is_match: bool
    best_distance: float | None
    best_face_location: tuple | None  # (top, right, bottom, left)
    number_of_detected_faces: int

    def __repr__(self) -> str:
        dist_str = f"{self.best_distance:.4f}" if self.best_distance is not None else "None"
        return (
            f"CandidateVerificationResult(is_match={self.is_match}, "
            f"best_distance={dist_str}, "
            f"best_face_location={self.best_face_location}, "
            f"number_of_detected_faces={self.number_of_detected_faces})"
        )


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_face(image_path: str) -> FaceEncodingResult:
    """
    Detect and encode the primary (largest) face in an image.

    Raises:
        FileNotFoundError: if image_path doesn't exist.
        ValueError: if no face is detected or image exceeds max dimensions.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(str(path)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image dimensions ({img.width}x{img.height}) exceed maximum allowed pixels ({MAX_IMAGE_PIXELS})")
            img = ImageOps.exif_transpose(img)
            image = np.array(img.convert("RGB"))
    except Exception as exc:
        if isinstance(exc, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to load image {image_path}: {exc}") from exc

    locations = face_recognition.face_locations(image, model="hog")  # use "cnn" if GPU available

    if not locations:
        raise ValueError(f"No face detected in {image_path}")

    # Pick the largest bounding box if multiple faces are present.
    def area(loc):
        top, right, bottom, left = loc
        return (bottom - top) * (right - left)

    primary_location = max(locations, key=area)
    encodings = face_recognition.face_encodings(image, known_face_locations=[primary_location])

    if not encodings:
        raise ValueError(f"Face detected but encoding failed for {image_path}")

    encoding = np.array(encodings[0], dtype=np.float64)
    if encoding.shape != (128,) or not np.all(np.isfinite(encoding)):
        raise ValueError(f"Generated face encoding is invalid or non-finite for {image_path}")

    return FaceEncodingResult(
        image_path=str(path),
        encoding=encoding,
        face_location=primary_location,
    )


def compare_faces(known_encoding: np.ndarray, candidate_encoding: np.ndarray, tolerance: float = 0.6) -> tuple[bool, float]:
    """
    Returns (is_match, distance). Lower distance = more similar.
    tolerance=0.6 is face_recognition's standard default threshold.
    """
    distance = float(np.linalg.norm(known_encoding - candidate_encoding))
    return distance <= tolerance, distance


def _load_image_from_input(candidate_image_input: bytes | str | Path | np.ndarray) -> np.ndarray | None:
    """
    Helper to load an image from bytes, a file path (str/Path), or an existing numpy array.
    Applies EXIF transpose normalization and pixel size safeguards.
    """
    if candidate_image_input is None:
        return None

    if isinstance(candidate_image_input, np.ndarray):
        if candidate_image_input.ndim != 3 or candidate_image_input.shape[2] != 3:
            return None
        if candidate_image_input.shape[0] * candidate_image_input.shape[1] > MAX_IMAGE_PIXELS:
            return None
        if candidate_image_input.dtype != np.uint8:
            try:
                candidate_image_input = candidate_image_input.astype(np.uint8)
            except Exception:
                return None
        return candidate_image_input

    if isinstance(candidate_image_input, bytes):
        if not candidate_image_input:
            return None
        try:
            with Image.open(io.BytesIO(candidate_image_input)) as img:
                if img.width * img.height > MAX_IMAGE_PIXELS:
                    return None
                img = ImageOps.exif_transpose(img)
                return np.array(img.convert("RGB"))
        except Exception:
            return None

    if isinstance(candidate_image_input, (str, Path)):
        path = Path(candidate_image_input)
        if not path.exists():
            return None
        try:
            with Image.open(str(path)) as img:
                if img.width * img.height > MAX_IMAGE_PIXELS:
                    return None
                img = ImageOps.exif_transpose(img)
                return np.array(img.convert("RGB"))
        except Exception:
            return None

    return None


def validate_input_encoding(input_encoding: Any) -> np.ndarray:
    """
    Validates that input_encoding is a 128-dimensional finite numeric vector.
    Raises ValueError if validation fails.
    """
    if input_encoding is None:
        raise ValueError("input_encoding cannot be None")

    if isinstance(input_encoding, list):
        try:
            input_encoding = np.array(input_encoding, dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"Failed to convert input_encoding list to numpy array: {exc}") from exc

    if not isinstance(input_encoding, np.ndarray):
        raise ValueError(f"input_encoding must be a numpy array or list, got {type(input_encoding).__name__}")

    if input_encoding.ndim != 1 or input_encoding.shape[0] != 128:
        raise ValueError(f"input_encoding must be 1-dimensional with length 128, got shape {input_encoding.shape}")

    if not np.all(np.isfinite(input_encoding)):
        raise ValueError("input_encoding contains NaN or infinite values")

    return input_encoding.astype(np.float64)


def validate_tolerance(tolerance: Any) -> float:
    """
    Validates that tolerance is a finite numeric value in sensible range [0.0, 2.0].
    Raises ValueError if validation fails.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError(f"tolerance must be a numeric float/int, got {type(tolerance).__name__}")

    tol_float = float(tolerance)
    if not math.isfinite(tol_float):
        raise ValueError(f"tolerance must be finite, got {tolerance}")

    if tol_float < 0.0 or tol_float > 2.0:
        raise ValueError(f"tolerance out of sensible range [0.0, 2.0]: {tolerance}")

    return tol_float


def verify_candidate_face(
    input_encoding: np.ndarray | list[float],
    candidate_image_input: bytes | str | Path | np.ndarray,
    tolerance: float = 0.6,
) -> CandidateVerificationResult:
    """
    Detects all faces in a candidate image, generates encodings for each,
    compares them against the input face encoding, and returns the strongest verified match.

    Args:
        input_encoding: 128-d face encoding of the input face (numpy array or list).
        candidate_image_input: Image bytes, file path, Path object, or numpy array.
        tolerance: Maximum Euclidean distance threshold for a positive match (default 0.6).

    Returns:
        CandidateVerificationResult detailing whether a match was found, the minimum
        face distance, the matching face location, and total detected faces count.
    """
    # 1. Validate inputs mathematically
    valid_encoding = validate_input_encoding(input_encoding)
    valid_tolerance = validate_tolerance(tolerance)

    # 2. Decode candidate image cleanly
    image = _load_image_from_input(candidate_image_input)
    if image is None:
        return CandidateVerificationResult(
            is_match=False,
            best_distance=None,
            best_face_location=None,
            number_of_detected_faces=0,
        )

    # 3. Detect ALL faces in candidate image
    try:
        locations = face_recognition.face_locations(image, model="hog")
    except Exception:
        return CandidateVerificationResult(
            is_match=False,
            best_distance=None,
            best_face_location=None,
            number_of_detected_faces=0,
        )

    if not locations:
        return CandidateVerificationResult(
            is_match=False,
            best_distance=None,
            best_face_location=None,
            number_of_detected_faces=0,
        )

    # 4. Generate encodings for ALL detected faces
    try:
        encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    except Exception:
        return CandidateVerificationResult(
            is_match=False,
            best_distance=None,
            best_face_location=None,
            number_of_detected_faces=len(locations),
        )

    if not encodings:
        return CandidateVerificationResult(
            is_match=False,
            best_distance=None,
            best_face_location=None,
            number_of_detected_faces=len(locations),
        )

    # 5. Compare each face against ORIGINAL input encoding and select minimum distance
    best_distance = float("inf")
    best_location = None

    for loc, enc in zip(locations, encodings):
        enc_arr = np.array(enc, dtype=np.float64)
        if enc_arr.shape != (128,) or not np.all(np.isfinite(enc_arr)):
            continue
        distance = float(np.linalg.norm(valid_encoding - enc_arr))
        if distance < best_distance:
            best_distance = distance
            best_location = loc

    if best_distance == float("inf"):
        return CandidateVerificationResult(
            is_match=False,
            best_distance=None,
            best_face_location=None,
            number_of_detected_faces=len(locations),
        )

    # 6. Verification gate: is_match iff minimum Euclidean distance <= tolerance
    is_match = best_distance <= valid_tolerance

    return CandidateVerificationResult(
        is_match=is_match,
        best_distance=best_distance,
        best_face_location=best_location,
        number_of_detected_faces=len(locations),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python face_id.py <image_path>")
        sys.exit(1)

    result = encode_face(sys.argv[1])
    print(f"Face detected at {result.face_location}")
    print(f"Image SHA-256: {result.image_sha256}")
    print(f"Encoding (first 5 dims): {result.encoding[:5]}")
