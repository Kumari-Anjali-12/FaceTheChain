# Face Identification & Blockchain Verification

Built for HH Goa 2026 Shortlisting Task 3.

## What it does

An end-to-end pipeline:

1. **Face ID** — detects and encodes a face from an input photo (`src/face_id.py`, using the `face_recognition` / dlib library).
2. **Web/social search** — runs a genuine reverse-image search against the web using Google Cloud Vision's Web Detection API, and picks the best candidate that looks like a social media post (`src/web_search.py`).
3. **Blockchain verification** — hashes the discovered post (URL + title + the original image's SHA-256) and writes that hash to a small Solidity smart contract, `RecordRegistry`, deployed either on a free local simulated chain or a public testnet. The pipeline then re-reads the on-chain record and confirms it matches, proving the record hasn't been tampered with since upload (`src/blockchain_verify.py`, `contracts/RecordRegistry.sol`).

`src/pipeline.py` wires all three stages together and prints a JSON summary (matched URL, on-chain hash, transaction hash, verification result).

### A note on consent

A tool that finds a stranger's social media presence from their face is a real privacy risk — this is the same category of technology that got Clearview AI banned in several countries. **Run this only on images you have the right to search: your own photo, or a teammate's with explicit consent for the demo.** Don't point it at a stranger without their permission.

## Which blockchain

By default, the pipeline runs against a **local, in-memory Ethereum-compatible chain** via `eth-tester` (`BLOCKCHAIN_MODE=local`). This needs no API keys, costs nothing, and is instant — good for a reliable screen recording.

It can also run against the **Sepolia public testnet** by setting `BLOCKCHAIN_MODE=testnet`, `WEB3_PROVIDER_URL` (an Infura/Alchemy RPC endpoint), and `PRIVATE_KEY` (a funded Sepolia test account — get test ETH from a Sepolia faucet). Same contract, same code path — only the environment variables change.

## How to run it

```bash
git clone <this-repo>
cd faceid-blockchain
pip install -r requirements.txt

# Stage 2 needs a Google Cloud Vision service-account key.
# Create a project at console.cloud.google.com, enable the Vision API,
# create a service account key, download the JSON.
cp .env.example .env
# edit .env: set GOOGLE_APPLICATION_CREDENTIALS to the key's path
export $(cat .env | xargs)

# First run only: this compiles the contract, which downloads solc
# (needs internet once, then it's cached locally).
python src/pipeline.py path/to/your_photo.jpg
```

Expected console output: face detection result → list of web matches → best social match URL → on-chain upload with tx hash → re-verification result (`Verified: True`) → JSON summary.

### Running the smoke tests for each stage individually

```bash
python src/face_id.py path/to/your_photo.jpg
python src/web_search.py path/to/your_photo.jpg
python src/blockchain_verify.py   # uploads + verifies a dummy record on the local chain
```

## Repo structure

```
├── src/
│   ├── face_id.py           # Stage 1: face detection + encoding
│   ├── web_search.py        # Stage 2: reverse image search (Google Vision Web Detection)
│   ├── blockchain_verify.py # Stage 3: hash, upload, and verify on-chain
│   └── pipeline.py          # orchestrator — run this for the full demo
├── contracts/
│   └── RecordRegistry.sol   # minimal append-only record registry contract
├── requirements.txt
├── .env.example
└── README.md
```

## Known limitations

- **Search coverage**: Google Vision's Web Detection only surfaces pages Google has indexed and where the image (or a near-duplicate) is publicly crawlable. It will miss posts behind login walls (most Instagram/Facebook content) or very recent uploads.
- **Face matching, not identity verification**: the pipeline finds pages containing a visually similar image; it does not cross-check that the matched post's face is a confirmed same-person match beyond what Vision's image similarity provides. For a stronger guarantee, cross-check the matched image's face encoding against the input using `face_id.compare_faces()`.
- **No takedown/on-chain content storage**: only a hash of the post's metadata is stored on-chain, not the image/content itself, to avoid needless duplication of third-party content and to keep gas costs low.
- **Contract has no owner/access control**: this is a minimal demo registry — any address can register a record, and once written a record cannot be edited or deleted (by design, for tamper-evidence), but there's no on-chain moderation.
- **Local chain state doesn't persist**: in `local` mode, the simulated chain resets every run — use `testnet` mode with a fixed `CONTRACT_ADDRESS` if you want persistent, publicly-checkable records across runs.
