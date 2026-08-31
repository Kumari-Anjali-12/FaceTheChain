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
from dataclasses import dataclass, field
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image


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
        ValueError: if no face is detected.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = face_recognition.load_image_file(str(path))
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

    return FaceEncodingResult(
        image_path=str(path),
        encoding=encodings[0],
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
    """
    if isinstance(candidate_image_input, np.ndarray):
        return candidate_image_input

    if isinstance(candidate_image_input, bytes):
        try:
            with Image.open(io.BytesIO(candidate_image_input)) as img:
                return np.array(img.convert("RGB"))
        except Exception:
            return None

    if isinstance(candidate_image_input, (str, Path)):
        path = Path(candidate_image_input)
        if not path.exists():
            return None
        try:
            return face_recognition.load_image_file(str(path))
        except Exception:
            return None

    return None


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
    if isinstance(input_encoding, list):
        input_encoding = np.array(input_encoding, dtype=np.float64)

    image = _load_image_from_input(candidate_image_input)
    if image is None:
        return CandidateVerificationResult(
            is_match=False,
            best_distance=None,
            best_face_location=None,
            number_of_detected_faces=0,
        )

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

    best_distance = float("inf")
    best_location = None

    for loc, enc in zip(locations, encodings):
        distance = float(np.linalg.norm(input_encoding - enc))
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

    is_match = best_distance <= tolerance

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
