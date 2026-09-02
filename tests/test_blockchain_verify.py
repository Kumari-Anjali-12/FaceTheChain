import os
import pytest
from hexbytes import HexBytes

from src.blockchain_verify import (
    normalize_hash,
    get_contract_artifacts,
    sha256_hex,
    hash_post_payload,
    upload_record,
    verify_record,
)


def test_normalize_hash_equivalence():
    """Test that 0x-prefixed, uppercase, bytes, and HexBytes normalize to lowercase hex without 0x."""
    h1 = "0xABC123"
    h2 = "abc123"
    h3 = HexBytes("0xABC123")
    h4 = bytes.fromhex("abc123")

    norm1 = normalize_hash(h1)
    norm2 = normalize_hash(h2)
    norm3 = normalize_hash(h3)
    norm4 = normalize_hash(h4)

    assert norm1 == "abc123"
    assert norm2 == "abc123"
    assert norm3 == "abc123"
    assert norm4 == "abc123"

    assert norm1 == norm2
    assert norm1 == norm3
    assert norm1 == norm4


def test_normalize_hash_inequality():
    """Test that different hashes are not considered equal."""
    h1 = "0xABC123"
    h2 = "0xDEF456"

    assert normalize_hash(h1) != normalize_hash(h2)


def test_lazy_contract_artifacts_loading():
    """Test that get_contract_artifacts returns valid ABI and bytecode."""
    abi, bytecode = get_contract_artifacts()
    assert isinstance(abi, list)
    assert len(abi) > 0
    assert isinstance(bytecode, str)
    assert len(bytecode) > 0


def test_sha256_hex_and_payload_hashing():
    """Test payload hashing logic."""
    raw_hash = sha256_hex(b"test data")
    assert len(raw_hash) == 64

    payload_hash = hash_post_payload(
        post_url="https://example.com/post/1",
        page_title="Example Title",
        image_sha256="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    )
    assert len(payload_hash) == 64


def test_local_blockchain_upload_and_verify(monkeypatch):
    """Test uploading a record to local eth-tester and verifying with normalized hash variants."""
    monkeypatch.setenv("BLOCKCHAIN_MODE", "local")

    record_id = "test-record-123"
    expected_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    record = upload_record(record_id, expected_hash)
    assert record.record_id == record_id
    assert record.data_hash == expected_hash.lower()
    assert record.uploader.startswith("0x")

    # Verify matching hash in lowercase without 0x
    assert verify_record(record_id, expected_hash) is True

    # Verify matching hash with 0x prefix and uppercase
    assert verify_record(record_id, f"0x{expected_hash.upper()}") is True

    # Verify non-matching hash fails
    wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    assert verify_record(record_id, wrong_hash) is False
