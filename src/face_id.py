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
from dataclasses import dataclass, field
from pathlib import Path

import face_recognition
import numpy as np


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


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python face_id.py <image_path>")
        sys.exit(1)

    result = encode_face(sys.argv[1])
    print(f"Face detected at {result.face_location}")
    print(f"Image SHA-256: {result.image_sha256}")
    print(f"Encoding (first 5 dims): {result.encoding[:5]}")
