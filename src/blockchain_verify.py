"""
blockchain_verify.py
---------------------
Stage 3 of the pipeline: take the discovered post's data, hash it, and
write that hash to a blockchain to create a tamper-evident record. Then
re-fetch the on-chain record and verify it matches — proving the record
hasn't been altered since upload.

Two modes, chosen by BLOCKCHAIN_MODE env var:

  "local"   (default) - spins up an in-memory Ethereum-compatible chain
             using eth-tester via web3.py. No API keys, no cost, no
             internet needed. Perfect for the demo recording and for
             judges to reproduce instantly.

  "testnet" - deploys/interacts with the same contract on a real public
             testnet (Sepolia) via an RPC URL (e.g. Infura/Alchemy).
             Requires WEB3_PROVIDER_URL and PRIVATE_KEY env vars.

Either way the on-chain artifact is the same: a mapping from a record
ID to (sha256 hash, uploader address, timestamp), stored via a tiny
Solidity contract (see contracts/RecordRegistry.sol).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

from web3 import Web3

CONTRACT_SOL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "contracts", "RecordRegistry.sol"
)
_SOLC_VERSION = "0.8.19"


def _compile_contract():
    """
    Compiles contracts/RecordRegistry.sol on the fly using py-solc-x.
    Caches the solc binary locally after the first run (needs internet
    the very first time you run the pipeline; none after that).
    """
    import solcx

    installed = solcx.get_installed_solc_versions()
    if not any(str(v) == _SOLC_VERSION for v in installed):
        solcx.install_solc(_SOLC_VERSION)
    solcx.set_solc_version(_SOLC_VERSION)

    compiled = solcx.compile_files(
        [CONTRACT_SOL_PATH],
        output_values=["abi", "bin"],
        solc_version=_SOLC_VERSION,
    )
    key = f"{CONTRACT_SOL_PATH}:RecordRegistry"
    return compiled[key]["abi"], compiled[key]["bin"]


CONTRACT_ABI, CONTRACT_BYTECODE = _compile_contract()


@dataclass
class OnChainRecord:
    record_id: str
    data_hash: str
    uploader: str
    timestamp: int
    tx_hash: str


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_post_payload(post_url: str, page_title: str | None, image_sha256: str) -> str:
    """
    Build a canonical payload for the discovered post and hash it.
    Combining the URL, title, and the original image's hash means the
    on-chain record ties the *specific* discovered post to the
    *specific* input face image.
    """
    payload = json.dumps(
        {"url": post_url, "title": page_title or "", "image_sha256": image_sha256},
        sort_keys=True,
    ).encode("utf-8")
    return sha256_hex(payload)


def _get_web3_and_account():
    mode = os.environ.get("BLOCKCHAIN_MODE", "local")

    if mode == "local":
        from eth_tester import EthereumTester
        from web3.providers.eth_tester import EthereumTesterProvider

        tester = EthereumTester()
        w3 = Web3(EthereumTesterProvider(tester))
        account = w3.eth.accounts[0]
        return w3, account, None  # no private key needed for eth-tester

    elif mode == "testnet":
        provider_url = os.environ["WEB3_PROVIDER_URL"]  # e.g. Infura Sepolia endpoint
        private_key = os.environ["PRIVATE_KEY"]
        w3 = Web3(Web3.HTTPProvider(provider_url))
        account = w3.eth.account.from_key(private_key).address
        return w3, account, private_key

    else:
        raise ValueError(f"Unknown BLOCKCHAIN_MODE: {mode}")


def deploy_or_get_contract(w3: Web3, account: str, private_key: str | None):
    """
    Deploys RecordRegistry if CONTRACT_ADDRESS isn't set, otherwise
    connects to the existing deployment. Deployment cost on the local
    tester is free/instant; on testnet it costs a small amount of test ETH.
    """
    existing = os.environ.get("CONTRACT_ADDRESS")
    if existing:
        return w3.eth.contract(address=Web3.to_checksum_address(existing), abi=CONTRACT_ABI)

    Contract = w3.eth.contract(abi=CONTRACT_ABI, bytecode=CONTRACT_BYTECODE)

    if private_key is None:
        tx_hash = Contract.constructor().transact({"from": account})
    else:
        nonce = w3.eth.get_transaction_count(account)
        tx = Contract.constructor().build_transaction(
            {"from": account, "nonce": nonce, "gas": 500_000}
        )
        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Contract deployed at {receipt.contractAddress}")
    return w3.eth.contract(address=receipt.contractAddress, abi=CONTRACT_ABI)


def upload_record(record_id: str, data_hash_hex: str) -> OnChainRecord:
    """
    Writes (record_id -> data_hash) on-chain.
    """
    w3, account, private_key = _get_web3_and_account()
    contract = deploy_or_get_contract(w3, account, private_key)

    record_id_bytes = Web3.keccak(text=record_id)
    data_hash_bytes = bytes.fromhex(data_hash_hex)

    if private_key is None:
        tx_hash = contract.functions.registerRecord(record_id_bytes, data_hash_bytes).transact(
            {"from": account}
        )
    else:
        nonce = w3.eth.get_transaction_count(account)
        tx = contract.functions.registerRecord(record_id_bytes, data_hash_bytes).build_transaction(
            {"from": account, "nonce": nonce, "gas": 200_000}
        )
        signed = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    stored_hash, uploader, timestamp = contract.functions.getRecord(record_id_bytes).call()

    return OnChainRecord(
        record_id=record_id,
        data_hash=stored_hash.hex(),
        uploader=uploader,
        timestamp=timestamp,
        tx_hash=receipt.transactionHash.hex(),
    )


def verify_record(record_id: str, expected_hash_hex: str) -> bool:
    """
    Re-reads the on-chain record and confirms it matches the expected hash.
    This is the "re-verifying the data against the on-chain record" step.
    """
    w3, account, private_key = _get_web3_and_account()
    contract = deploy_or_get_contract(w3, account, private_key)

    record_id_bytes = Web3.keccak(text=record_id)
    stored_hash, uploader, timestamp = contract.functions.getRecord(record_id_bytes).call()

    return stored_hash.hex().lower() == expected_hash_hex.lower()


if __name__ == "__main__":
    # Smoke test on the local simulated chain.
    os.environ.setdefault("BLOCKCHAIN_MODE", "local")
    demo_hash = hash_post_payload("https://example.com/post/1", "Demo Post", "abc123")
    record = upload_record(record_id=f"demo-{int(time.time())}", data_hash_hex=demo_hash)
    print(record)
    print("Verified:", verify_record(record.record_id, demo_hash))
