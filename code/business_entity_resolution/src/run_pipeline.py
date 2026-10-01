#!/usr/bin/env python3
"""
Official Test Set Submission Generator — EXP-007b Full-Pool 50/68-Feature Ensemble
with Global Country-Wide Bipartite Mutual Exclusion.

Generates and validates BOTH required official submission files:
  - output/matching_results.tsv   (1,732,544 S1 rows: source1_entity_id \t matched_entity_ids)
  - output/candidate_pairs.tsv    (1,732,544 S1 rows: source1_entity_id \t candidate_entity_ids)

Key Innovations in EXP-007b over EXP-006:
  1. Upgraded Preprocessor & Blocker (99.92% Validation Blocker Recall):
     - Fixed 4 Indic script bugs (Bengali/Odia ya, Gurmukhi Tippi/Addak/sha, zero-width joiners).
     - Standalone 'private'/'pvt' & 'publiclimited', duplicate word/bigram collapsing, OCR digit fixes.
     - Alphanumeric number-word splitting ('188BIS' -> '188 bis', '146Kailash' -> '146 kailash').
  2. 50 Pairwise & Sibling-Cluster Features + 18 Group Context = 68 Stage-2 Features:
     - Pure-street similarity, 3+ digit building number rescue, sibling address cluster consensus,
       clean missing-address group separator, French legal/region tokens, exact initials acronym rescue,
       synthetic generic-word substitution rescue, shifted primary street numbers, 3-level legal conflict.
  3. Global Country-Wide Bipartite Mutual Exclusion:
     - Enforces the 100.000% Ground-Truth Single-Parent Law across all S1 entities in each country
       (France: 259,452 S1, US: 663,106 S1, India: 809,986 S1) before applying country-calibrated
       dual-threshold + drop-off ratio rules.
"""

import gc
import heapq
import math
import os
import subprocess
import sys
import time
from array import array
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from code.business_entity_resolution.src.blocking import extract_blocking_keys
from code.business_entity_resolution.src.data_loader import EntityRecord
from code.business_entity_resolution.src.exp005_postproc_reranker import STAGE2_CONTEXT_FEATURE_NAMES, build_stage2_context_features
from code.business_entity_resolution.src.features import FEATURE_NAMES, extract_features_for_s1_group
from code.business_entity_resolution.src.preprocessor import (
    clean_address,
    clean_business_name,
    extract_name_variants,
    phonetic_skeleton_word,
)

ALL_STAGE2_FEATURE_NAMES_EXP007 = list(FEATURE_NAMES) + STAGE2_CONTEXT_FEATURE_NAMES

# Read-only worker globals populated before fork() and frozen with gc.freeze()
_G_C_IDX: Dict[str, array] = {}
_G_TOKEN_DF: Dict[str, int] = {}
_G_CAND_EIDS: List[str] = []
_G_CAND_NAMES: List[str] = []
_G_CAND_ADDRS: List[str] = []
_G_COUNTRY: str = ""
_G_TOP_K: int = 30

# Worker-local models initialized AFTER fork() inside _init_worker()
_W_S1_LGB_PURE = None
_W_S1_LGB_EXT = None
_W_S1_XGB_PURE = None
_W_S1_XGB_PURE_END: int = 0
_W_S1_XGB_EXT = None
_W_S1_XGB_EXT_END: int = 0
_W_S2_LGB = None
_W_S2_XGB = None
_W_S2_XGB_END: int = 0
_W_XGB_MOD = None

_PREFIX_BASE_W = {
    "ns:": 3.5,
    "nc:": 3.5,
    "ps:": 2.8,
    "af:": 2.8,
    "sh:": 2.2,
    "an:": 2.2,
    "nt:": 1.5,
    "aw:": 1.5,
    "pc:": 1.5,
    "ph:": 1.25,
    "ng:": 0.35,
    "np:": 0.45,
}

# Country-calibrated EXP-007b thresholds: (tau_top1, tau_sec, min_ratio, rescue_consensus_tau, dedup_margin)
_COUNTRY_THRESHOLDS = {
    "US": (0.58, 0.56, 0.68, 0.56, 0.04),
    "India": (0.56, 0.60, 0.68, 0.54, 0.04),
    "France": (0.58, 0.58, 0.68, 0.56, 0.04),
}


def _clear_all_caches() -> None:
    clean_business_name.cache_clear()
    extract_name_variants.cache_clear()
    clean_address.cache_clear()
    phonetic_skeleton_word.cache_clear()


def _get_xgb_end_round(bst) -> int:
    best_it = bst.attr("best_iteration")
    if best_it is not None:
        return int(best_it) + 1
    return bst.num_boosted_rounds()


def _extract_s1_keys_worker(batch: List[Tuple[str, str, str]]) -> Set[str]:
    """Extracts the union of blocking keys for a batch of S1 tuples in _G_COUNTRY."""
    country = _G_COUNTRY
    keys: Set[str] = set()
    for s1_id, b_name, b_addr in batch:
        rec = EntityRecord(s1_id, b_name, b_addr, country)
        keys.update(extract_blocking_keys(rec))
    _clear_all_caches()
    return keys


def _query_entity_int(s1_record: EntityRecord) -> List[int]:
    """
    Retrieves top-K integer candidate indices for a single S1 record using
    Multi-Channel Cross-Field Synergy & Guaranteed Quotas over `array('I')` postings.
    """
    c_idx = _G_C_IDX
    if not c_idx:
        return []

    name_scores: Dict[int, float] = defaultdict(float)
    skel_scores: Dict[int, float] = defaultdict(float)
    addr_scores: Dict[int, float] = defaultdict(float)

    for key in extract_blocking_keys(s1_record):
        postings = c_idx.get(key)
        if not postings:
            continue
        df = len(postings)
        prefix = key[:3]
        base_w = _PREFIX_BASE_W.get(prefix, 0.45)
        w = base_w / (math.log(2.0 + df) ** 2.0)

        if prefix in ("af:", "an:", "aw:", "pc:"):
            for cid in postings:
                addr_scores[cid] += w
        elif prefix == "nc:":
            for cid in postings:
                name_scores[cid] += w
                skel_scores[cid] += w
                addr_scores[cid] += w
        else:
            for cid in postings:
                name_scores[cid] += w
                if prefix in ("ps:", "ph:"):
                    skel_scores[cid] += w

    if not name_scores and not addr_scores:
        return []

    k = _G_TOP_K
    all_cands = set(name_scores.keys()) | set(addr_scores.keys())
    if len(all_cands) <= k:
        return [
            cid
            for cid, _ in sorted(
                ((c, name_scores.get(c, 0.0) + addr_scores.get(c, 0.0)) for c in all_cands),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

    synergy_scores: Dict[int, float] = {}
    for cid in all_cands:
        ns = name_scores.get(cid, 0.0)
        ads = addr_scores.get(cid, 0.0)
        bonus = 0.50 if (ns > 0.0 and ads > 0.0) else 0.0
        synergy_scores[cid] = ns + 0.6 * ads + bonus

    top_syn = [cid for cid, _ in heapq.nlargest(k, synergy_scores.items(), key=lambda x: x[1])]
    top_name = (
        [cid for cid, _ in heapq.nlargest(12, name_scores.items(), key=lambda x: x[1])]
        if name_scores
        else []
    )
    top_skel = (
        [cid for cid, _ in heapq.nlargest(10, skel_scores.items(), key=lambda x: x[1])]
        if skel_scores
        else []
    )
    top_addr = (
        [cid for cid, _ in heapq.nlargest(10, addr_scores.items(), key=lambda x: x[1])]
        if addr_scores
        else []
    )

    selected: List[int] = []
    seen: Set[int] = set()

    def add_from(lst: List[int], limit: int) -> None:
        c = 0
        for cid in lst:
            if cid not in seen:
                seen.add(cid)
                selected.append(cid)
                c += 1
                if c >= limit or len(selected) >= k:
                    break

    q_syn = max(1, int(k * 0.58))
    q_name = max(1, int(k * 0.16))
    q_skel = max(1, int(k * 0.12))
    q_addr = max(1, k - q_syn - q_name - q_skel)

    add_from(top_syn, q_syn)
    add_from(top_name, q_name)
    add_from(top_skel, q_skel)
    add_from(top_addr, q_addr)
    if len(selected) < k:
        add_from(top_syn, k - len(selected))

    return selected


def _init_worker(model_dir: str) -> None:
    """
    Loads the pre-trained EXP-007b Stage-1 (50 features: Pure + Extended LightGBM & GPU-trained XGBoost)
    and Stage-2 (68 features: LightGBM + XGBoost Meta-Rerankers) inside each worker process
    AFTER fork() with single-threaded execution (nthread=1).
    """
    global _W_S1_LGB_PURE, _W_S1_LGB_EXT
    global _W_S1_XGB_PURE, _W_S1_XGB_PURE_END, _W_S1_XGB_EXT, _W_S1_XGB_EXT_END
    global _W_S2_LGB, _W_S2_XGB, _W_S2_XGB_END, _W_XGB_MOD

    os.environ["OMP_NUM_THREADS"] = "1"
    import lightgbm as lgb
    import xgboost as xgb

    _W_XGB_MOD = xgb

    # Stage-1 50-Feature Models (Cross-Fold Pure + Extended Ensemble)
    _W_S1_LGB_PURE = lgb.Booster(model_file=os.path.join(model_dir, "lgbm_exp007_pure_fold0.txt"))
    _W_S1_LGB_PURE.params["num_threads"] = 1

    _W_S1_LGB_EXT = lgb.Booster(model_file=os.path.join(model_dir, "lgbm_exp007_ext_fold1.txt"))
    _W_S1_LGB_EXT.params["num_threads"] = 1

    _W_S1_XGB_PURE = xgb.Booster()
    _W_S1_XGB_PURE.load_model(os.path.join(model_dir, "xgb_exp007_pure_fold2.json"))
    _W_S1_XGB_PURE.set_param({"nthread": 1, "device": "cpu"})
    _W_S1_XGB_PURE_END = _get_xgb_end_round(_W_S1_XGB_PURE)

    _W_S1_XGB_EXT = xgb.Booster()
    _W_S1_XGB_EXT.load_model(os.path.join(model_dir, "xgb_exp007_ext_fold3.json"))
    _W_S1_XGB_EXT.set_param({"nthread": 1, "device": "cpu"})
    _W_S1_XGB_EXT_END = _get_xgb_end_round(_W_S1_XGB_EXT)

    # Stage-2 68-Feature Group-Context Models (0.99975 correlation with 5-Fold OOF)
    _W_S2_LGB = lgb.Booster(model_file=os.path.join(model_dir, "stage2_lgb_exp007_fold0.txt"))
    _W_S2_LGB.params["num_threads"] = 1

    _W_S2_XGB = xgb.Booster()
    _W_S2_XGB.load_model(os.path.join(model_dir, "stage2_xgb_exp007_fold2.json"))
    _W_S2_XGB.set_param({"nthread": 1, "device": "cpu"})
    _W_S2_XGB_END = _get_xgb_end_round(_W_S2_XGB)


def _predict_batch_worker(
    s1_batch: List[Tuple[str, str, str]]
) -> List[Tuple[str, str, str]]:
    """
    Processes a batch of (s1_id, business_name, business_address) records for _G_COUNTRY:
      1. Queries top-30 integer candidate indices via `_query_entity_int`.
      2. Extracts the 50 EXP-007b pairwise, structural, numeric-multiset, and sibling-cluster features.
      3. Runs Stage-1 (Pure + Ext LightGBM & XGBoost) and Stage-2 (68-feature Meta-Reranker).
      4. Returns (s1_id, cand_ids_csv, scored_cands_str) where:
         - cand_ids_csv: comma-separated candidate IDs sorted by final probability (for candidate_pairs.tsv)
         - scored_cands_str: comma-separated `cid:prob` for candidates with prob >= 0.44
           (for country-wide Bipartite Mutual Exclusion & dual-threshold selection).
    """
    country = _G_COUNTRY
    c_df = _G_TOKEN_DF
    eids = _G_CAND_EIDS
    names = _G_CAND_NAMES
    addrs = _G_CAND_ADDRS

    active_s1_indices: List[int] = []
    batch_cids: List[List[str]] = []
    group_lengths: List[int] = []
    X_parts: List[np.ndarray] = []

    results: List[Tuple[str, str, str]] = [None] * len(s1_batch)  # type: ignore

    for idx, (s1_id, b_name, b_addr) in enumerate(s1_batch):
        s1_rec = EntityRecord(s1_id, b_name, b_addr, country)
        cand_ints = _query_entity_int(s1_rec)
        if not cand_ints:
            results[idx] = (s1_id, "", "")
            continue

        c_recs = [EntityRecord(eids[ci], names[ci], addrs[ci], country) for ci in cand_ints]
        cids_str = [eids[ci] for ci in cand_ints]
        feats = extract_features_for_s1_group(s1_rec, c_recs, country_token_df=c_df)

        active_s1_indices.append(idx)
        batch_cids.append(cids_str)
        group_lengths.append(len(cand_ints))
        X_parts.append(feats)

    if not X_parts:
        _clear_all_caches()
        return results

    X_base = np.vstack(X_parts)

    # Stage-1 Predictions (50 Features: 55% Pure + 45% Extended)
    p_lgb_p = _W_S1_LGB_PURE.predict(X_base, num_threads=1).astype(np.float32)
    p_lgb_e = _W_S1_LGB_EXT.predict(X_base, num_threads=1).astype(np.float32)
    p_lgb = 0.55 * p_lgb_p + 0.45 * p_lgb_e

    dmat_s1 = _W_XGB_MOD.DMatrix(X_base, feature_names=FEATURE_NAMES)
    p_xgb_p = _W_S1_XGB_PURE.predict(dmat_s1, iteration_range=(0, _W_S1_XGB_PURE_END)).astype(np.float32)
    p_xgb_e = _W_S1_XGB_EXT.predict(dmat_s1, iteration_range=(0, _W_S1_XGB_EXT_END)).astype(np.float32)
    p_xgb = 0.55 * p_xgb_p + 0.45 * p_xgb_e

    # Stage-2 Group-Context Features (68 Features = 50 Pairwise + 18 Group Context)
    X_stage2 = build_stage2_context_features(X_base, p_lgb, p_xgb, group_lengths, blend_w_lgb=0.55)
    p_s2_lgb = _W_S2_LGB.predict(X_stage2, num_threads=1).astype(np.float32)
    dmat_s2 = _W_XGB_MOD.DMatrix(X_stage2, feature_names=ALL_STAGE2_FEATURE_NAMES_EXP007)
    p_s2_xgb = _W_S2_XGB.predict(dmat_s2, iteration_range=(0, _W_S2_XGB_END)).astype(np.float32)

    p_s1_blend = 0.55 * p_lgb + 0.45 * p_xgb
    p_s2_ens = 0.50 * p_s2_lgb + 0.50 * p_s2_xgb
    p_final = 0.80 * p_s2_ens + 0.20 * p_s1_blend

    offset = 0
    for active_pos, s1_idx in enumerate(active_s1_indices):
        s1_id = s1_batch[s1_idx][0]
        cids = batch_cids[active_pos]
        length = group_lengths[active_pos]
        end = offset + length
        g_probs = p_final[offset:end]
        offset = end

        pairs = sorted(
            ((cids[i], float(g_probs[i])) for i in range(length)),
            key=lambda x: x[1],
            reverse=True,
        )
        cand_csv = ",".join(cid for cid, _ in pairs)
        scored_str = ",".join(f"{cid}:{prob:.5f}" for cid, prob in pairs if prob >= 0.44)
        results[s1_idx] = (s1_id, cand_csv, scored_str)

    _clear_all_caches()
    return results


def load_country_s1_tuples(path: str, target_country: str) -> List[Tuple[str, str, str]]:
    """Loads (s1_id, name, address) tuples for target_country from test_source1.tsv."""
    rows: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not line.endswith(target_country + "\n") and not line.endswith(target_country + "\r\n"):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3].strip() == target_country:
                rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return rows


def stream_and_index_chunk(
    test_dir: str,
    country: str,
    s1_chunk: List[Tuple[str, str, str]],
    cached_tdf: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, array], Dict[str, int], List[str], List[str], List[str]]:
    """
    1. Extracts `needed_keys` in parallel from `s1_chunk` (<= 270,000 S1 entities).
    2. Streams test_source2.tsv and test_source3.tsv for `country`, indexing only keys
       present in `needed_keys` into `array('I')` 32-bit unsigned integer posting lists
       with immediate in-loop pruning (`2500 / 1000 / 250`).
    """
    global _G_COUNTRY
    _G_COUNTRY = country
    t0 = time.time()

    key_batches = [s1_chunk[i : i + 4000] for i in range(0, len(s1_chunk), 4000)]
    needed_keys: Set[str] = set()
    with Pool(processes=6) as key_pool:
        for kset in key_pool.imap_unordered(_extract_s1_keys_worker, key_batches):
            needed_keys.update(kset)
    del key_batches
    gc.collect()
    print(
        f"    [1/3] Extracted {len(needed_keys):,} unique query keys from {len(s1_chunk):,} "
        f"{country} S1 entities in {time.time() - t0:.1f}s",
        flush=True,
    )

    t_scan = time.time()
    c_idx: Dict[str, array] = {}
    build_tdf = cached_tdf is None
    c_tdf: Dict[str, int] = defaultdict(int) if build_tdf else cached_tdf  # type: ignore

    high_prec_limit = 2500   # ns:, ps:, nc:, sh:, af:, an:
    mid_prec_limit = 1000    # nt:, ph:, aw:, pc:
    sub_limit = 250          # ng:, np:

    cand_eids: List[str] = []
    cand_names: List[str] = []
    cand_addrs: List[str] = []
    doc_cnt = 0

    for fname in ("test_source2.tsv", "test_source3.tsv"):
        fpath = os.path.join(test_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                if not line.endswith(country + "\n") and not line.endswith(country + "\r\n"):
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4 or parts[3].strip() != country:
                    continue

                eid = parts[0].strip()
                b_name = parts[1].strip()
                b_addr = parts[2].strip()
                doc_cnt += 1

                rec = EntityRecord(eid, b_name, b_addr, country)
                keys = extract_blocking_keys(rec)

                if build_tdf:
                    for key in keys:
                        if key.startswith("nt:"):
                            c_tdf[key[3:]] += 1

                matched_keys = [k for k in keys if k in needed_keys]
                if matched_keys:
                    cid_int = len(cand_eids)
                    added_any = False
                    for key in matched_keys:
                        arr = c_idx.get(key)
                        if arr is None:
                            arr = array("I")
                            c_idx[key] = arr

                        if key.startswith(("ng:", "np:")):
                            lim = sub_limit
                        elif key.startswith(("ns:", "ps:", "nc:", "sh:", "af:", "an:")):
                            lim = high_prec_limit
                        else:
                            lim = mid_prec_limit

                        if len(arr) >= lim:
                            del c_idx[key]
                            needed_keys.discard(key)
                        else:
                            arr.append(cid_int)
                            added_any = True

                    if added_any:
                        cand_eids.append(eid)
                        cand_names.append(b_name)
                        cand_addrs.append(b_addr)

                if doc_cnt % 500_000 == 0:
                    _clear_all_caches()
                    print(
                        f"      [{country}] Streamed {doc_cnt:,} candidates "
                        f"(active keys: {len(c_idx):,}, retained cands: {len(cand_eids):,}, "
                        f"{time.time() - t_scan:.1f}s)...",
                        flush=True,
                    )

    needed_keys.clear()
    _clear_all_caches()

    if build_tdf:
        ref_pool_size = 234_392.0
        scale = min(1.0, ref_pool_size / float(max(1, doc_cnt)))
        if scale < 0.95:
            for tok, raw_df in list(c_tdf.items()):
                c_tdf[tok] = max(1, int(round(raw_df * scale)))
        c_tdf = dict(c_tdf)

    gc.collect()
    print(
        f"    [2/3] Indexed {doc_cnt:,} {country} candidates -> {len(c_idx):,} active `array('I')` keys, "
        f"{len(cand_eids):,} reachable candidates in {time.time() - t_scan:.1f}s",
        flush=True,
    )
    return c_idx, c_tdf, cand_eids, cand_names, cand_addrs


def apply_country_bipartite_dedup(
    country: str,
    raw_scores_path: str,
    match_ckpt: str,
) -> Tuple[int, int]:
    """
    Applies Global Country-Wide Bipartite Mutual Exclusion across all S1 entities of `country`:
      1. First pass over `raw_scores_path` finds `cand_best_prob[cid]` and `cand_best_s1[cid]`.
      2. Second pass filters each S1 entity's candidates by bipartite ownership and applies
         the country-calibrated dual-threshold + drop-off ratio rule, writing `match_ckpt`.
    Returns (non_singleton_count, total_links).
    """
    t0 = time.time()
    tau_top1, tau_sec, min_ratio, rescue_consensus_tau, dedup_margin = _COUNTRY_THRESHOLDS[country]

    cand_best_prob: Dict[str, float] = {}
    cand_best_s1: Dict[str, str] = {}

    with open(raw_scores_path, "r", encoding="utf-8") as f:
        for line in f:
            s1_id, tab, rest = line.partition("\t")
            if not tab:
                continue
            s_str = rest.strip()
            if not s_str:
                continue
            for item in s_str.split(","):
                cid, _, p_str = item.partition(":")
                prob = float(p_str)
                if prob > cand_best_prob.get(cid, -1.0):
                    cand_best_prob[cid] = prob
                    cand_best_s1[cid] = s1_id

    non_singleton_count = 0
    total_links = 0
    dedup_removed = 0
    tmp_out = match_ckpt + ".bipartite.tmp"

    with open(raw_scores_path, "r", encoding="utf-8") as fin, open(tmp_out, "w", encoding="utf-8") as fout:
        for line in fin:
            s1_id, tab, rest = line.partition("\t")
            if not tab:
                continue
            s_str = rest.strip()
            if not s_str:
                fout.write(f"{s1_id}\t\n")
                continue

            valid_pairs: List[Tuple[str, float]] = []
            for item in s_str.split(","):
                cid, _, p_str = item.partition(":")
                prob = float(p_str)
                if cand_best_s1.get(cid) == s1_id or (cand_best_prob[cid] - prob) <= dedup_margin:
                    valid_pairs.append((cid, prob))
                elif prob >= tau_sec:
                    dedup_removed += 1

            if not valid_pairs:
                fout.write(f"{s1_id}\t\n")
                continue

            p1 = valid_pairs[0][1]
            p2 = valid_pairs[1][1] if len(valid_pairs) > 1 else 0.0
            is_non_singleton = (p1 >= tau_top1) or (
                p1 >= rescue_consensus_tau and p2 >= rescue_consensus_tau
            )
            if not is_non_singleton:
                fout.write(f"{s1_id}\t\n")
                continue

            matched = [valid_pairs[0][0]]
            for cid, prob in valid_pairs[1:]:
                if prob >= tau_sec and (prob / (p1 + 1e-6)) >= min_ratio:
                    matched.append(cid)
                else:
                    break

            non_singleton_count += 1
            total_links += len(matched)
            fout.write(f"{s1_id}\t{','.join(matched)}\n")

    os.replace(tmp_out, match_ckpt)
    print(
        f"    [Bipartite Mutual Exclusion — {country}] Pruned {dedup_removed:,} cross-S1 duplicate claims "
        f"in {time.time() - t0:.1f}s -> Non-Singletons: {non_singleton_count:,}, Total Links: {total_links:,}",
        flush=True,
    )
    return non_singleton_count, total_links


def process_country(
    country: str,
    test_dir: str,
    model_dir: str,
    ckpt_dir: str,
    num_workers: int = 6,
    chunk_size: int = 260_000,
) -> Tuple[str, str]:
    """
    Runs end-to-end blocking, 50-feature extraction, hierarchical Stage-1 + Stage-2 inference,
    and Global Country-Wide Bipartite Mutual Exclusion for a single country, saving both:
      - `matching_exp007_{country}.tsv`
      - `candidates_exp007_{country}.tsv`
    to `ckpt_dir`.
    """
    global _G_C_IDX, _G_TOKEN_DF, _G_CAND_EIDS, _G_CAND_NAMES, _G_CAND_ADDRS, _G_COUNTRY

    match_ckpt = os.path.join(ckpt_dir, f"matching_exp007_{country}.tsv")
    cand_ckpt = os.path.join(ckpt_dir, f"candidates_exp007_{country}.tsv")
    raw_scores_ckpt = os.path.join(ckpt_dir, f"raw_scores_exp007_{country}.tsv")

    s1_rows = load_country_s1_tuples(os.path.join(test_dir, "test_source1.tsv"), country)
    expected_count = len(s1_rows)

    if os.path.isfile(match_ckpt) and os.path.isfile(cand_ckpt):
        with open(match_ckpt, "r", encoding="utf-8") as fm, open(cand_ckpt, "r", encoding="utf-8") as fc:
            m_lines = sum(1 for _ in fm)
            c_lines = sum(1 for _ in fc)
        if m_lines == expected_count and c_lines == expected_count:
            print(
                f"\n✅ [{country}] EXP-007b Checkpoints already complete ({m_lines:,} S1 rows). Skipping!",
                flush=True,
            )
            return match_ckpt, cand_ckpt

    t_c0 = time.time()
    print("\n" + "=" * 92)
    print(f"🌍 PROCESSING COUNTRY (EXP-007b): {country} ({expected_count:,} Source 1 Entities)")
    print("=" * 92, flush=True)

    tmp_scores = raw_scores_ckpt + ".tmp"
    tmp_cand = cand_ckpt + ".tmp"
    done_ids: Set[str] = set()
    processed = 0

    if os.path.isfile(tmp_scores) and os.path.isfile(tmp_cand):
        valid_s_lines: List[str] = []
        valid_c_lines: List[str] = []
        s_map: Dict[str, str] = {}
        with open(tmp_scores, "r", encoding="utf-8") as fs_in:
            for s_line in fs_in:
                if s_line.endswith("\n") and "\t" in s_line:
                    sid, _, _ = s_line.partition("\t")
                    if sid:
                        s_map[sid] = s_line
        with open(tmp_cand, "r", encoding="utf-8") as fc_in:
            for c_line in fc_in:
                if c_line.endswith("\n") and "\t" in c_line:
                    sid, _, _ = c_line.partition("\t")
                    if sid and sid in s_map and sid not in done_ids:
                        done_ids.add(sid)
                        valid_s_lines.append(s_map[sid])
                        valid_c_lines.append(c_line)
        del s_map
        if done_ids:
            with open(tmp_scores, "w", encoding="utf-8") as fs_out:
                fs_out.writelines(valid_s_lines)
            with open(tmp_cand, "w", encoding="utf-8") as fc_out:
                fc_out.writelines(valid_c_lines)
            processed = len(done_ids)
            print(
                f"  🔄 Resuming {country} from {processed:,}/{expected_count:,} already-scored S1 rows!",
                flush=True,
            )
        del valid_s_lines, valid_c_lines

    remaining_rows = [r for r in s1_rows if r[0] not in done_ids] if done_ids else s1_rows
    del s1_rows, done_ids
    gc.collect()

    chunks = [
        remaining_rows[i : i + chunk_size]
        for i in range(0, len(remaining_rows), chunk_size)
    ]
    del remaining_rows
    gc.collect()

    cached_tdf: Optional[Dict[str, int]] = None

    for chunk_idx, s1_chunk in enumerate(chunks, 1):
        print(
            f"\n  --- [{country}] S1 Chunk {chunk_idx}/{len(chunks)} ({len(s1_chunk):,} S1 Entities) ---",
            flush=True,
        )
        c_idx, cached_tdf, cand_eids, cand_names, cand_addrs = stream_and_index_chunk(
            test_dir=test_dir,
            country=country,
            s1_chunk=s1_chunk,
            cached_tdf=cached_tdf,
        )

        _G_C_IDX = c_idx
        _G_TOKEN_DF = cached_tdf
        _G_CAND_EIDS = cand_eids
        _G_CAND_NAMES = cand_names
        _G_CAND_ADDRS = cand_addrs
        _G_COUNTRY = country

        batch_size = 500
        batches = [s1_chunk[i : i + batch_size] for i in range(0, len(s1_chunk), batch_size)]
        print(
            f"    [3/3] Scoring {len(s1_chunk):,} S1 entities across {len(batches)} batches "
            f"using {num_workers} parallel workers (maxtasksperchild=12)...",
            flush=True,
        )

        t_inf = time.time()
        chunk_processed = 0
        open_mode = "a" if processed > 0 else "w"

        gc.freeze()
        with open(tmp_scores, open_mode, encoding="utf-8") as fs, open(
            tmp_cand, open_mode, encoding="utf-8"
        ) as fc:
            with Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(model_dir,),
                maxtasksperchild=12,
            ) as pool:
                for batch_idx, batch_out in enumerate(pool.imap(_predict_batch_worker, batches), 1):
                    s_lines = []
                    c_lines = []
                    for s1_id, cand_csv, scored_str in batch_out:
                        s_lines.append(f"{s1_id}\t{scored_str}\n")
                        c_lines.append(f"{s1_id}\t{cand_csv}\n")
                    fs.writelines(s_lines)
                    fc.writelines(c_lines)
                    fs.flush()
                    fc.flush()
                    processed += len(batch_out)
                    chunk_processed += len(batch_out)

                    if batch_idx % 30 == 0 or chunk_processed == len(s1_chunk):
                        elapsed = time.time() - t_inf
                        rate = chunk_processed / max(0.1, elapsed)
                        eta = (expected_count - processed) / max(1.0, rate)
                        print(
                            f"      [{country}] {processed:>7,}/{expected_count:,} ({processed/expected_count*100:>5.1f}%) | "
                            f"Speed: {rate:,.0f} S1/s | ETA: {eta/60:.1f}m",
                            flush=True,
                        )
        gc.unfreeze()

        _G_C_IDX.clear()
        _G_CAND_EIDS.clear()
        _G_CAND_NAMES.clear()
        _G_CAND_ADDRS.clear()
        del c_idx, cand_eids, cand_names, cand_addrs, batches
        _clear_all_caches()
        gc.collect()

    _G_TOKEN_DF = {}
    del cached_tdf, chunks
    gc.collect()

    os.replace(tmp_scores, raw_scores_ckpt)
    os.replace(tmp_cand, cand_ckpt)

    non_singleton_count, total_links = apply_country_bipartite_dedup(
        country=country,
        raw_scores_path=raw_scores_ckpt,
        match_ckpt=match_ckpt,
    )

    print(
        f"  🎉 Completed {country} in {(time.time() - t_c0)/60:.2f} min! "
        f"(Predicted Singletons: {(1.0 - non_singleton_count/expected_count)*100:.2f}%, "
        f"Non-Singletons: {non_singleton_count:,}, Total Matched Links: {total_links:,})",
        flush=True,
    )
    return match_ckpt, cand_ckpt


def main():
    test_dir = os.path.join(PROJECT_ROOT, "dataset/test")
    model_dir = os.path.join(PROJECT_ROOT, "output/models")
    output_dir = os.path.join(PROJECT_ROOT, "output")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    t_total = time.time()
    print("=" * 92)
    print("🚀 GENERATING OFFICIAL TEST SUBMISSION (EXP-007b 50/68-FEATURE ENSEMBLE + BIPARTITE DEDUP)")
    print("=" * 92, flush=True)

    # Process smallest country first (France -> US -> India)
    countries = ["France", "US", "India"]
    for country in countries:
        process_country(
            country=country,
            test_dir=test_dir,
            model_dir=model_dir,
            ckpt_dir=ckpt_dir,
            num_workers=6,
            chunk_size=260_000,
        )

    # Merge country checkpoints in exact test_source1.tsv order
    print("\n" + "=" * 92)
    print("📦 MERGING COUNTRY CHECKPOINTS INTO output/matching_results.tsv & output/candidate_pairs.tsv...")
    print("=" * 92, flush=True)

    ordered_s1_ids: List[str] = []
    with open(os.path.join(test_dir, "test_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if line.strip():
                ordered_s1_ids.append(line.split("\t", 1)[0].strip())

    # 1. Write output/matching_results.tsv
    match_map: Dict[str, str] = {}
    for country in countries:
        m_path = os.path.join(ckpt_dir, f"matching_exp007_{country}.tsv")
        with open(m_path, "r", encoding="utf-8") as f:
            for line in f:
                s1_id, _, rest = line.partition("\t")
                match_map[s1_id] = rest

    final_matching_path = os.path.join(output_dir, "matching_results.tsv")
    with open(final_matching_path, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in ordered_s1_ids:
            fm.write(f"{s1_id}\t{match_map.get(s1_id, '\n')}")

    del match_map
    gc.collect()
    print(f"  --> Wrote {len(ordered_s1_ids):,} rows to {final_matching_path}")

    # 2. Write output/candidate_pairs.tsv
    cand_map: Dict[str, str] = {}
    for country in countries:
        c_path = os.path.join(ckpt_dir, f"candidates_exp007_{country}.tsv")
        with open(c_path, "r", encoding="utf-8") as f:
            for line in f:
                s1_id, _, rest = line.partition("\t")
                cand_map[s1_id] = rest

    final_candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    with open(final_candidate_path, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in ordered_s1_ids:
            fc.write(f"{s1_id}\t{cand_map.get(s1_id, '\n')}")

    del cand_map
    gc.collect()
    print(f"  --> Wrote {len(ordered_s1_ids):,} rows to {final_candidate_path}")

    # 3. Run official submission validator (first standard mode on both files, then --check-ids)
    print("\n🔍 Running Official Submission Validator (matching_results.tsv + candidate_pairs.tsv)...")
    val_cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "utils/validate_submission.py"),
        "--matching",
        final_matching_path,
        "--candidate",
        final_candidate_path,
        "--test-dir",
        test_dir,
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    print("\n🔍 Running Official Submission Validator with --check-ids on matching_results.tsv...")
    val_id_cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "utils/validate_submission.py"),
        "--matching",
        final_matching_path,
        "--test-dir",
        test_dir,
        "--check-ids",
    ]
    res_id = subprocess.run(val_id_cmd, capture_output=True, text=True)
    print(res_id.stdout)
    if res_id.stderr:
        print(res_id.stderr)

    print(
        f"\n🏆 Total Official EXP-007b Submission Generation & Validation Time: "
        f"{(time.time() - t_total)/60:.2f} minutes!"
    )


if __name__ == "__main__":
    main()
