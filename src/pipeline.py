"""
pipeline.py
-----------
End-to-end orchestrator: face scan -> web search -> candidate image download ->
candidate face verification -> canonical evidence hashing -> blockchain verification ->
tamper detection test.

Run this file directly for CLI / demo execution:
    python src/pipeline.py path/to/your_photo.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

try:
    from src.blockchain_verify import OnChainRecord, upload_record, verify_record
    from src.face_id import CandidateVerificationResult, encode_face, verify_candidate_face
    from src.web_search import WebMatch, download_candidate_image, search_by_image
except ImportError:
    from blockchain_verify import OnChainRecord, upload_record, verify_record
    from face_id import CandidateVerificationResult, encode_face, verify_candidate_face
    from web_search import WebMatch, download_candidate_image, search_by_image


@dataclass
class PipelineResult:
    status: str
    input_image_path: str
    input_image_sha256: Optional[str] = None
    candidates_evaluated: int = 0
    verified_candidate: Optional[Dict[str, Any]] = None
    canonical_evidence: Optional[Dict[str, Any]] = None
    evidence_hash: Optional[str] = None
    record_id: Optional[str] = None
    on_chain_record: Optional[Dict[str, Any]] = None
    original_verified: bool = False
    tamper_detection_successful: bool = False
    error_message: Optional[str] = None


def build_canonical_evidence(
    input_image_sha256: str,
    candidate_page_url: Optional[str],
    candidate_image_url: Optional[str],
    candidate_image_sha256: str,
    match_type: str,
    web_search_score: Optional[float],
    verification_result: CandidateVerificationResult,
    tolerance: float,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs a deterministic canonical evidence dictionary.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    location = (
        list(verification_result.best_face_location)
        if verification_result.best_face_location
        else None
    )

    return {
        "schema_version": "1.0",
        "input_image_sha256": input_image_sha256,
        "candidate_page_url": candidate_page_url or "",
        "candidate_image_url": candidate_image_url or "",
        "candidate_image_sha256": candidate_image_sha256,
        "match_type": match_type,
        "web_search_score": web_search_score,
        "face_distance": round(float(verification_result.best_distance), 4)
        if verification_result.best_distance is not None
        else None,
        "face_tolerance": float(tolerance),
        "is_face_match": bool(verification_result.is_match),
        "detected_face_count": int(verification_result.number_of_detected_faces),
        "matched_face_location": location,
        "timestamp": timestamp,
    }


def canonicalize_and_hash(evidence: Dict[str, Any]) -> Tuple[str, bytes]:
    """
    Serializes evidence dict to deterministic UTF-8 JSON bytes and calculates its SHA-256 hash.
    """
    canonical_bytes = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    sha256_hex = hashlib.sha256(canonical_bytes).hexdigest()
    return sha256_hex, canonical_bytes


def generate_deterministic_record_id(input_sha: str, candidate_sha: str) -> str:
    """
    Generates a deterministic 32-character hex record ID based on input and candidate image hashes.
    """
    source = f"faceid:{input_sha}:{candidate_sha}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]


def run_pipeline(
    image_path: str, tolerance: float = 0.6, max_candidates: int = 10
) -> PipelineResult:
    """
    Executes the end-to-end FaceTheChain pipeline.
    """
    print(f"[1/4] Detecting and encoding face in {image_path} ...")
    try:
        face_result = encode_face(image_path)
    except Exception as exc:
        return PipelineResult(
            status="BLOCKCHAIN_ERROR",
            input_image_path=image_path,
            error_message=f"Failed to encode input face: {exc}",
        )

    input_sha = face_result.image_sha256
    print(f"      Face found at {face_result.face_location}")
    print(f"      Input Image SHA-256: {input_sha}")

    print("\n[2/4] Searching web/social media for candidate matches ...")
    try:
        matches = search_by_image(image_path)
    except Exception as exc:
        return PipelineResult(
            status="BLOCKCHAIN_ERROR",
            input_image_path=image_path,
            input_image_sha256=input_sha,
            error_message=f"Web search failed: {exc}",
        )

    if not matches:
        print("      No web candidates found.")
        return PipelineResult(
            status="NO_CANDIDATE_FOUND",
            input_image_path=image_path,
            input_image_sha256=input_sha,
        )

    print(f"      Discovered {len(matches)} raw candidate match(es).")
    candidates_to_eval = matches[:max_candidates]

    downloaded_any = False
    detected_faces_any = False
    candidates_eval_count = 0

    best_match_evidence: Optional[Dict[str, Any]] = None
    best_candidate_sha: Optional[str] = None
    best_match: Optional[WebMatch] = None
    best_verify_result: Optional[CandidateVerificationResult] = None

    print("\n[3/4] Downloading candidate images and verifying face matches ...")
    for idx, match in enumerate(candidates_to_eval, start=1):
        candidates_eval_count += 1
        img_url = match.direct_image_url or match.url
        print(f"      Candidate [{idx}/{len(candidates_to_eval)}]: {img_url} (type={match.match_type})")

        img_bytes = download_candidate_image(img_url)
        if img_bytes is None:
            print("        -> Download / SSRF validation failed. Skipping.")
            continue

        downloaded_any = True
        cand_sha = hashlib.sha256(img_bytes).hexdigest()

        verify_res = verify_candidate_face(
            face_result.encoding, img_bytes, tolerance=tolerance
        )

        if verify_res.number_of_detected_faces > 0:
            detected_faces_any = True

        print(
            f"        -> Detected faces: {verify_res.number_of_detected_faces}, "
            f"Best distance: {verify_res.best_distance}, Match: {verify_res.is_match}"
        )

        if verify_res.is_match:
            print("        -> VERIFIED FACE MATCH FOUND!")
            best_match = match
            best_candidate_sha = cand_sha
            best_verify_result = verify_res
            best_match_evidence = build_canonical_evidence(
                input_image_sha256=input_sha,
                candidate_page_url=match.url,
                candidate_image_url=match.direct_image_url,
                candidate_image_sha256=cand_sha,
                match_type=match.match_type,
                web_search_score=match.score,
                verification_result=verify_res,
                tolerance=tolerance,
            )
            break

    if best_match_evidence is None or best_candidate_sha is None or best_match is None or best_verify_result is None:
        if not downloaded_any:
            status = "DOWNLOAD_FAILED"
        elif not detected_faces_any:
            status = "NO_FACES_DETECTED"
        else:
            status = "NO_VERIFIED_FACE_MATCH"

        return PipelineResult(
            status=status,
            input_image_path=image_path,
            input_image_sha256=input_sha,
            candidates_evaluated=candidates_eval_count,
        )

    print("\n[4/4] Hashing canonical evidence & registering on-chain ...")
    evidence_hash, _ = canonicalize_and_hash(best_match_evidence)
    record_id = generate_deterministic_record_id(input_sha, best_candidate_sha)
    print(f"      Record ID: {record_id}")
    print(f"      Evidence SHA-256 Hash: {evidence_hash}")

    try:
        on_chain = upload_record(record_id=record_id, data_hash_hex=evidence_hash)
        print(f"      Uploaded to blockchain. Tx Hash: {on_chain.tx_hash}")
        original_verified = verify_record(record_id, evidence_hash)
        print(f"      Integrity verification (Original): {original_verified}")

        # --- Tamper Detection Test ---
        tampered_evidence = dict(best_match_evidence)
        tampered_evidence["candidate_image_sha256"] = "0" * 64
        tampered_hash, _ = canonicalize_and_hash(tampered_evidence)
        tampered_verification = verify_record(record_id, tampered_hash)
        tamper_successful = not tampered_verification
        print(f"      Tamper Test (Modified Evidence Hash Verified): {tampered_verification}")
        print(f"      Tamper Detection Result: {'PASSED (Tamper Caught)' if tamper_successful else 'FAILED'}")

        on_chain_dict = {
            "record_id": on_chain.record_id,
            "data_hash": on_chain.data_hash,
            "uploader": on_chain.uploader,
            "timestamp": on_chain.timestamp,
            "tx_hash": on_chain.tx_hash,
        }

        verified_candidate_dict = {
            "page_url": best_match.url,
            "image_url": best_match.direct_image_url,
            "image_sha256": best_candidate_sha,
            "match_type": best_match.match_type,
            "web_search_score": best_match.score,
            "face_distance": best_verify_result.best_distance,
            "face_location": best_verify_result.best_face_location,
            "number_of_detected_faces": best_verify_result.number_of_detected_faces,
        }

        return PipelineResult(
            status="VERIFIED_MATCH_REGISTERED",
            input_image_path=image_path,
            input_image_sha256=input_sha,
            candidates_evaluated=candidates_eval_count,
            verified_candidate=verified_candidate_dict,
            canonical_evidence=best_match_evidence,
            evidence_hash=evidence_hash,
            record_id=record_id,
            on_chain_record=on_chain_dict,
            original_verified=original_verified,
            tamper_detection_successful=tamper_successful,
        )

    except Exception as exc:
        return PipelineResult(
            status="BLOCKCHAIN_ERROR",
            input_image_path=image_path,
            input_image_sha256=input_sha,
            candidates_evaluated=candidates_eval_count,
            error_message=f"Blockchain operation failed: {exc}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face -> web search -> candidate verification -> blockchain verification pipeline"
    )
    parser.add_argument(
        "image_path", help="Path to input face image (use explicit consent photo for demo)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Face matching distance tolerance threshold (default: 0.6)",
    )
    args = parser.parse_args()

    res = run_pipeline(args.image_path, tolerance=args.tolerance)
    print("\n=== FINAL PIPELINE RESULT ===")
    print(json.dumps(asdict(res), indent=2))
    if res.status != "VERIFIED_MATCH_REGISTERED":
        sys.exit(1)
