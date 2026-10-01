"""
Data loading module for Business Entity Resolution Challenge.
Loads TSV files for Source 1, Source 2, Source 3, and Ground Truth.
"""

import os
from typing import Dict, List, Optional, Set, Tuple


class EntityRecord:
    __slots__ = ("entity_id", "business_name", "business_address", "country", "source")

    def __init__(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: str,
    ):
        self.entity_id = entity_id
        self.business_name = business_name
        self.business_address = business_address
        self.country = country
        self.source = entity_id.split("-")[0] if "-" in entity_id else "UNKNOWN"

    def __repr__(self) -> str:
        return (
            f"EntityRecord(id={self.entity_id}, name={self.business_name!r}, "
            f"address={self.business_address!r}, country={self.country!r})"
        )


def load_source_tsv(file_path: str) -> Dict[str, EntityRecord]:
    """
    Loads a source TSV file (train_source*.tsv or test_source*.tsv) into a dict of EntityRecord.
    Expected columns: entity_id, business_name, business_address, country.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    records: Dict[str, EntityRecord] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return records
        
        headers = [h.strip() for h in header_line.strip("\r\n").split("\t")]
        id_idx = headers.index("entity_id") if "entity_id" in headers else 0
        name_idx = headers.index("business_name") if "business_name" in headers else 1
        addr_idx = headers.index("business_address") if "business_address" in headers else 2
        ctry_idx = headers.index("country") if "country" in headers else 3

        for line_num, line in enumerate(f, start=2):
            line = line.strip("\r\n")
            if not line:
                continue
            cols = line.split("\t")
            entity_id = cols[id_idx].strip() if len(cols) > id_idx else ""
            name = cols[name_idx].strip() if len(cols) > name_idx else ""
            addr = cols[addr_idx].strip() if len(cols) > addr_idx else ""
            country = cols[ctry_idx].strip() if len(cols) > ctry_idx else ""

            if entity_id:
                records[entity_id] = EntityRecord(
                    entity_id=entity_id,
                    business_name=name,
                    business_address=addr,
                    country=country,
                )
    return records


def load_ground_truth_tsv(file_path: str) -> Dict[str, Set[str]]:
    """
    Loads train_ground_truth.tsv.
    Expected columns: source1_entity_id, matched_entity_ids (comma-separated).
    Returns mapping: source1_entity_id -> set of matched entity IDs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    gt: Dict[str, Set[str]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return gt
        
        for line in f:
            line = line.strip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                s1_id = parts[0].strip()
                matched_raw = parts[1].strip()
                if matched_raw:
                    gt[s1_id] = set(m.strip() for m in matched_raw.split(",") if m.strip())
                else:
                    gt[s1_id] = set()
            elif len(parts) == 1:
                s1_id = parts[0].strip()
                gt[s1_id] = set()
    return gt
