// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title RecordRegistry
/// @notice Minimal tamper-evident record store: maps a record ID to a
///         data hash, the uploader address, and the block timestamp.
///         Once written, a record cannot be altered (no update/delete
///         function is exposed) — only re-read for verification.
contract RecordRegistry {
    struct Record {
        bytes32 dataHash;
        address uploader;
        uint256 timestamp;
        bool exists;
    }

    mapping(bytes32 => Record) private records;

    event RecordRegistered(bytes32 indexed recordId, bytes32 dataHash, address indexed uploader, uint256 timestamp);

    function registerRecord(bytes32 recordId, bytes32 dataHash) external {
        require(!records[recordId].exists, "Record already exists");
        records[recordId] = Record({
            dataHash: dataHash,
            uploader: msg.sender,
            timestamp: block.timestamp,
            exists: true
        });
        emit RecordRegistered(recordId, dataHash, msg.sender, block.timestamp);
    }

    function getRecord(bytes32 recordId) external view returns (bytes32, address, uint256) {
        Record memory r = records[recordId];
        require(r.exists, "Record not found");
        return (r.dataHash, r.uploader, r.timestamp);
    }
}
