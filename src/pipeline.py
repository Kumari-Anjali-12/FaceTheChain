"""
pipeline.py
-----------
End-to-end orchestrator: face scan -> web/social search -> blockchain
verification. Run this file directly for the demo recording.

    python src/pipeline.py path/to/your_photo.jpg

Ethics/consent note: point this at your own photo (or a teammate's,
with their explicit consent) for the demo. See README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from face_id import encode_face
from web_search import search_by_image, pick_best_social_match
from blockchain_verify import hash_post_payload, upload_record, verify_record


def run_pipeline(image_path: str) -> dict:
    print(f"[1/3] Detecting and encoding face in {image_path} ...")
    face_result = encode_face(image_path)
    print(f"      Face found at {face_result.face_location}")
    print(f"      Image SHA-256: {face_result.image_sha256}")

    print("\n[2/3] Searching the web for matching social media content ...")
    matches = search_by_image(image_path)
    if not matches:
        print("      No matches found. Try a different / higher-quality photo.")
        sys.exit(1)

    best = pick_best_social_match(matches)
    print(f"      Found {len(matches)} candidate match(es).")
    print(f"      Best social match: {best.url} (type={best.match_type}, score={best.score})")

    print("\n[3/3] Hashing the discovered post and writing it on-chain ...")
    data_hash = hash_post_payload(best.url, best.page_title, face_result.image_sha256)
    record_id = f"faceid-{int(time.time())}"

    on_chain = upload_record(record_id=record_id, data_hash_hex=data_hash)
    print(f"      Uploaded. Tx hash: {on_chain.tx_hash}")
    print(f"      On-chain record: hash={on_chain.data_hash} uploader={on_chain.uploader} ts={on_chain.timestamp}")

    print("\nRe-fetching on-chain record to verify integrity ...")
    is_verified = verify_record(record_id, data_hash)
    print(f"      Verified: {is_verified}")

    summary = {
        "image_path": image_path,
        "image_sha256": face_result.image_sha256,
        "matched_post_url": best.url,
        "matched_post_title": best.page_title,
        "record_id": record_id,
        "on_chain_hash": on_chain.data_hash,
        "tx_hash": on_chain.tx_hash,
        "uploader": on_chain.uploader,
        "timestamp": on_chain.timestamp,
        "verified": is_verified,
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face -> social search -> blockchain verification pipeline")
    parser.add_argument("image_path", help="Path to the input face image (use your own photo for the demo)")
    args = parser.parse_args()

    run_pipeline(args.image_path)
