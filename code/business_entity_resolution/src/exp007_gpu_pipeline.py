#!/usr/bin/env python3
"""
EXP-007: 50-Feature Full-Pool Disambiguation + GPU Multi-Model Ensemble + Global Bipartite Mutual Exclusion.

Upgrades over EXP-006:
  1. Preprocessor & Blocker Upgrades:
     - Alphanumeric number-word splitting ('188BIS' -> '188 bis', '146Kailash' -> '146 kailash', '2Sheela' -> '2 sheela')
     - Non-word-boundary digit extraction (?<!\d)\d+(?!\d) catching '2081d', '16a'
     - Leading honorific stripping ('Mr Shakti Global' -> 'shakti global', 'Smt Dovavantageumbra' -> 'dovavantageumbra')
     - French legal structures (SARL, SAS/SASU, EURL, SCI, SA, SNC) & region/street stopwords
     - Strict synthetic coined-alias filter (rejecting 'Vision', '49Ers', '#kmacademy')
  2. 10 New Pairwise & Sibling-Cluster Features (50 Base Features + 18 Group Context = 68 Stage-2 Features):
     - pure_street_sort_sim, pure_street_set_sim, min_pure_street_words
     - shared_3plus_digit_nums, unmatched_2plus_digit_nums_cnt
     - s1_same_name_match_addr_cnt, s1_same_name_diff_addr_cnt (92.7% TP vs 30.8% TP missing-address separator)
     - cand_addr_cluster_size, sim_to_top_addr_cand_addr (sibling cluster consensus)
     - cred_or_short_code_conflict (US medical/professional credential & short-acronym conflict detector)
  3. 5-Fold GroupKFold GPU XGBoost + LightGBM Stage-1 & Stage-2 Ensemble + Bipartite Mutual Exclusion.
"""

import gc
import os
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from code.business_entity_resolution.src.blocking import GENERIC_CORP_WORDS, PURE_STOP_WORDS
from code.business_entity_resolution.src.data_loader import EntityRecord, load_ground_truth_tsv, load_source_tsv
from code.business_entity_resolution.src.exp005_postproc_reranker import (
    STAGE2_CONTEXT_FEATURE_NAMES,
    build_stage2_context_features,
    evaluate_dual_threshold_rule,
)
from code.business_entity_resolution.src.features import FEATURE_NAMES, extract_features_for_s1_group
from code.business_entity_resolution.src.preprocessor import clean_address, clean_business_name, extract_name_variants
from utils.metrics import evaluate_predictions

ALL_STAGE2_FEATURE_NAMES_EXP007 = list(FEATURE_NAMES) + STAGE2_CONTEXT_FEATURE_NAMES

_G_S1_MAP: Dict[str, EntityRecord] = {}
_G_CAND_POOL: Dict[str, EntityRecord] = {}
_G_COUNTRY_DF: Dict[str, Dict[str, int]] = {}


def _feat_worker_exp007(batch: List[Tuple[str, List[str]]]) -> List[np.ndarray]:
    """Extracts 50-feature matrices for a batch of (s1_id, cand_ids) groups."""
    out = []
    for sid, cids in batch:
        if not cids:
            out.append(np.empty((0, len(FEATURE_NAMES)), dtype=np.float32))
            continue
        s1_rec = _G_S1_MAP[sid]
        c_recs = [_G_CAND_POOL[cid] for cid in cids]
        c_df = _G_COUNTRY_DF.get(s1_rec.country)
        feats = extract_features_for_s1_group(s1_rec, c_recs, country_token_df=c_df)
        out.append(feats)
    clean_business_name.cache_clear()
    extract_name_variants.cache_clear()
    clean_address.cache_clear()
    return out


def evaluate_with_bipartite_dedup(
    s1_pred_groups: Dict[str, List[Tuple[str, float]]],
    val_gt: Dict[str, Set[str]],
    tau_top1: float,
    tau_sec: float,
    min_ratio: float = 0.0,
    rescue_consensus_tau: float = 0.68,
    dedup_margin: float = 0.0,
) -> Dict[str, float]:
    """
    Applies the dual-threshold rule followed by Global Bipartite Mutual Exclusion:
    Since 100.000% of ground-truth S2/S3 records belong to at most ONE S1 parent entity,
    if a candidate S2/S3 ID is claimed by multiple S1 entities, only the S1 entity with
    the highest probability P(S1, cand) retains the candidate (unless within dedup_margin).
    """
    # 1. Find best S1 probability for each candidate across all S1 groups
    cand_best_prob: Dict[str, float] = {}
    cand_best_s1: Dict[str, str] = {}
    for sid, pairs in s1_pred_groups.items():
        for cid, prob in pairs:
            if prob > cand_best_prob.get(cid, -1.0):
                cand_best_prob[cid] = prob
                cand_best_s1[cid] = sid

    # 2. Filter each S1 group's candidates by bipartite ownership before thresholding
    preds: Dict[str, Set[str]] = {}
    for sid, pairs in s1_pred_groups.items():
        if not pairs:
            preds[sid] = set()
            continue
        valid_pairs = [
            (cid, prob)
            for cid, prob in pairs
            if cand_best_s1.get(cid) == sid or (cand_best_prob[cid] - prob) <= dedup_margin
        ]
        if not valid_pairs:
            preds[sid] = set()
            continue

        p1 = valid_pairs[0][1]
        p2 = valid_pairs[1][1] if len(valid_pairs) > 1 else 0.0
        is_non_singleton = (p1 >= tau_top1) or (p1 >= rescue_consensus_tau and p2 >= rescue_consensus_tau)
        if not is_non_singleton:
            preds[sid] = set()
            continue

        matched = {valid_pairs[0][0]}
        for cid, prob in valid_pairs[1:]:
            if prob >= tau_sec and (prob / (p1 + 1e-6)) >= min_ratio:
                matched.add(cid)
            else:
                break
        preds[sid] = matched

    return evaluate_predictions(val_gt, preds)


def main():
    global _G_S1_MAP, _G_CAND_POOL, _G_COUNTRY_DF

    val_dir = os.path.join(PROJECT_ROOT, "dataset/val")
    model_dir = os.path.join(PROJECT_ROOT, "output/models")
    os.makedirs(model_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 98)
    print("🚀 RUNNING EXP-007: 50-FEATURE FULL-POOL (1.03 CRORE) GPU ENSEMBLE & BIPARTITE MUTUAL EXCLUSION")
    print("=" * 98, flush=True)

    val_s1 = load_source_tsv(os.path.join(val_dir, "val_source1.tsv"))
    val_gt = load_ground_truth_tsv(os.path.join(val_dir, "val_ground_truth.tsv"))

    meta_pkl = os.path.join(model_dir, "val_meta_exp006_fullpool.pkl")
    old_val_npz = os.path.join(model_dir, "val_matrix_exp006_fullpool.npz")
    old_ext_npz = os.path.join(model_dir, "ext_matrix_exp006_fullpool.npz")
    cand_pkl = os.path.join(model_dir, "val_fullpool_cand_records.pkl")
    exp007_val_npz = os.path.join(model_dir, "val_matrix_exp007_fullpool.npz")

    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)
    row_s1_ids: List[str] = meta["row_s1_ids"]
    row_cand_ids: List[str] = meta["row_cand_ids"]
    group_lengths: List[int] = meta["group_lengths"]

    d_old_val = np.load(old_val_npz)
    X_val_old, y_val = d_old_val["X_val"], d_old_val["y_val"]

    if os.path.isfile(exp007_val_npz):
        print(f"\n[Stage 1] Loading cached 50-Feature Full-Pool Matrix from {exp007_val_npz}...")
        X_val = np.load(exp007_val_npz)["X_val"]
        print(f"  Loaded X_val={X_val.shape} in {time.time() - t_start:.2f}s", flush=True)
    else:
        t_f0 = time.time()
        print("\n[Stage 1] Extracting 50-Feature Full-Pool Matrix across 20,000 Val S1 Entities (699,953 pairs)...")
        with open(cand_pkl, "rb") as f:
            cand_pool: Dict[str, EntityRecord] = pickle.load(f)

        # Build country token DF scaled to 234,392 reference scale
        c_df_raw: Dict[str, Dict[str, int]] = {"India": defaultdict(int), "US": defaultdict(int)}
        c_counts = {"India": 0, "US": 0}
        for rec in cand_pool.values():
            c = rec.country
            if c in c_df_raw:
                c_counts[c] += 1
                _, strip_n = clean_business_name(rec.business_name)
                for tok in set(strip_n.split()):
                    if len(tok) >= 2 and tok not in PURE_STOP_WORDS:
                        c_df_raw[c][tok] += 1
        clean_business_name.cache_clear()

        norm_df: Dict[str, Dict[str, int]] = {}
        for c in ("India", "US"):
            scale = 234_392.0 / max(1.0, float(c_counts[c]))
            norm_df[c] = {tok: max(1, int(round(cnt * scale))) for tok, cnt in c_df_raw[c].items()}

        # Reconstruct ordered (s1_id, cand_ids) groups matching row_s1_ids and group_lengths
        groups: List[Tuple[str, List[str]]] = []
        offset = 0
        for length in group_lengths:
            sid = row_s1_ids[offset]
            cids = row_cand_ids[offset:offset + length]
            groups.append((sid, cids))
            offset += length

        _G_S1_MAP = val_s1
        _G_CAND_POOL = cand_pool
        _G_COUNTRY_DF = norm_df

        batches = [groups[i:i + 500] for i in range(0, len(groups), 500)]
        X_parts: List[np.ndarray] = []
        gc.freeze()
        with Pool(processes=8) as pool:
            for batch_res in pool.imap(_feat_worker_exp007, batches):
                X_parts.extend(batch_res)
        gc.unfreeze()

        X_val = np.vstack(X_parts)
        # Preserve exact 1.03-Crore full-pool IDF columns (10 and 11) from EXP-006
        X_val[:, 10] = X_val_old[:, 10]
        X_val[:, 11] = X_val_old[:, 11]

        _G_S1_MAP = {}
        _G_CAND_POOL = {}
        _G_COUNTRY_DF = {}
        del cand_pool, X_parts
        gc.collect()

        np.savez(exp007_val_npz, X_val=X_val)
        print(f"  Extracted & saved 50-Feature X_val={X_val.shape} in {time.time() - t_f0:.2f}s!", flush=True)

    # Load X_ext (835,969 x 40) and pad columns 40..49 with NaN for extended model diversity
    d_ext = np.load(old_ext_npz)
    X_ext_40, y_ext = d_ext["X_ext"], d_ext["y_ext"]
    X_ext_50 = np.full((len(y_ext), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    X_ext_50[:, :40] = X_ext_40
    del X_ext_40, X_val_old
    gc.collect()

    # 2. Train Stage-1 50-Feature GPU XGBoost + LightGBM Models (5-Fold GroupKFold OOF)
    print("\n[Stage 2] Training Stage-1 50-Feature GPU XGBoost & LightGBM (Pure Val + Extended Full-Pool)...")
    unique_val_s1 = list(val_s1.keys())
    s1_to_fold = {sid: (idx % 5) for idx, sid in enumerate(unique_val_s1)}
    row_folds = np.array([s1_to_fold[sid] for sid in row_s1_ids], dtype=np.int32)

    oof_xgb_pure = np.zeros(len(y_val), dtype=np.float32)
    oof_lgb_pure = np.zeros(len(y_val), dtype=np.float32)
    oof_xgb_ext = np.zeros(len(y_val), dtype=np.float32)
    oof_lgb_ext = np.zeros(len(y_val), dtype=np.float32)

    xgb_params_pure = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 8,
        "learning_rate": 0.045,
        "subsample": 0.85,
        "colsample_bytree": 0.78,
        "min_child_weight": 18.0,
        "reg_alpha": 1.0,
        "reg_lambda": 2.5,
        "max_bin": 256,
        "seed": 42,
    }

    xgb_params_ext = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 9,
        "learning_rate": 0.045,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_weight": 22.0,
        "reg_alpha": 1.2,
        "reg_lambda": 3.0,
        "max_bin": 256,
        "seed": 107,
    }

    lgb_params_pure = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.045,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 35,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.78,
        "reg_alpha": 1.0,
        "reg_lambda": 2.5,
        "n_jobs": 12,
        "random_state": 42,
        "verbose": -1,
    }

    lgb_params_ext = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.045,
        "num_leaves": 79,
        "max_depth": 9,
        "min_child_samples": 45,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.2,
        "reg_lambda": 3.0,
        "n_jobs": 12,
        "random_state": 107,
        "verbose": -1,
    }

    s1_importances = {f: 0.0 for f in FEATURE_NAMES}

    for fold in range(5):
        t_f = time.time()
        val_mask = (row_folds == fold)
        trn_mask = ~val_mask
        X_va, y_va = X_val[val_mask], y_val[val_mask]
        X_tr_pure, y_tr_pure = X_val[trn_mask], y_val[trn_mask]

        # A. Pure 50-Feature GPU XGBoost (Zero NaNs on all 50 features)
        dtr_xp = xgb.QuantileDMatrix(X_tr_pure, label=y_tr_pure, feature_names=FEATURE_NAMES, max_bin=256)
        dva_xp = xgb.QuantileDMatrix(X_va, label=y_va, ref=dtr_xp, feature_names=FEATURE_NAMES, max_bin=256)
        bst_p = xgb.train(
            xgb_params_pure,
            dtr_xp,
            num_boost_round=800,
            evals=[(dva_xp, "val")],
            early_stopping_rounds=45,
            verbose_eval=False,
        )
        oof_xgb_pure[val_mask] = bst_p.predict(dva_xp, iteration_range=(0, bst_p.best_iteration + 1))
        bst_p.save_model(os.path.join(model_dir, f"xgb_exp007_pure_fold{fold}.json"))
        del dtr_xp, dva_xp, bst_p

        # B. Pure 50-Feature LightGBM
        dtr_lp = lgb.Dataset(X_tr_pure, label=y_tr_pure, feature_name=FEATURE_NAMES)
        dva_lp = lgb.Dataset(X_va, label=y_va, reference=dtr_lp, feature_name=FEATURE_NAMES)
        ml_p = lgb.train(
            lgb_params_pure,
            dtr_lp,
            num_boost_round=750,
            valid_sets=[dva_lp],
            callbacks=[lgb.early_stopping(stopping_rounds=45, verbose=False)],
        )
        oof_lgb_pure[val_mask] = ml_p.predict(X_va, num_iteration=ml_p.best_iteration)
        ml_p.save_model(os.path.join(model_dir, f"lgbm_exp007_pure_fold{fold}.txt"), num_iteration=ml_p.best_iteration)
        for fn, gv in zip(FEATURE_NAMES, ml_p.feature_importance("gain")):
            s1_importances[fn] += float(gv) / 10.0
        del dtr_lp, dva_lp, ml_p

        # C. Extended 1.40M-Row GPU XGBoost + LightGBM
        X_tr_ext = np.vstack([X_ext_50, X_tr_pure])
        y_tr_ext = np.concatenate([y_ext, y_tr_pure])
        w_tr_ext = np.concatenate([
            np.full(len(y_ext), 0.55, dtype=np.float32),
            np.ones(len(y_tr_pure), dtype=np.float32),
        ])

        dtr_xe = xgb.QuantileDMatrix(X_tr_ext, label=y_tr_ext, weight=w_tr_ext, feature_names=FEATURE_NAMES, max_bin=256)
        dva_xe = xgb.QuantileDMatrix(X_va, label=y_va, ref=dtr_xe, feature_names=FEATURE_NAMES, max_bin=256)
        bst_e = xgb.train(
            xgb_params_ext,
            dtr_xe,
            num_boost_round=800,
            evals=[(dva_xe, "val")],
            early_stopping_rounds=45,
            verbose_eval=False,
        )
        oof_xgb_ext[val_mask] = bst_e.predict(dva_xe, iteration_range=(0, bst_e.best_iteration + 1))
        bst_e.save_model(os.path.join(model_dir, f"xgb_exp007_ext_fold{fold}.json"))
        del dtr_xe, dva_xe, bst_e

        dtr_le = lgb.Dataset(X_tr_ext, label=y_tr_ext, weight=w_tr_ext, feature_name=FEATURE_NAMES)
        dva_le = lgb.Dataset(X_va, label=y_va, reference=dtr_le, feature_name=FEATURE_NAMES)
        ml_e = lgb.train(
            lgb_params_ext,
            dtr_le,
            num_boost_round=750,
            valid_sets=[dva_le],
            callbacks=[lgb.early_stopping(stopping_rounds=45, verbose=False)],
        )
        oof_lgb_ext[val_mask] = ml_e.predict(X_va, num_iteration=ml_e.best_iteration)
        ml_e.save_model(os.path.join(model_dir, f"lgbm_exp007_ext_fold{fold}.txt"), num_iteration=ml_e.best_iteration)
        for fn, gv in zip(FEATURE_NAMES, ml_e.feature_importance("gain")):
            s1_importances[fn] += float(gv) / 10.0
        del dtr_le, dva_le, ml_e, X_tr_ext, y_tr_ext, w_tr_ext
        gc.collect()

        print(f"  [Stage-1 Fold {fold + 1}/5] Trained 4 GPU XGBoost + LightGBM models in {time.time() - t_f:.1f}s", flush=True)

    del X_ext_50, y_ext
    gc.collect()

    oof_xgb = 0.55 * oof_xgb_pure + 0.45 * oof_xgb_ext
    oof_lgb = 0.55 * oof_lgb_pure + 0.45 * oof_lgb_ext
    np.save(os.path.join(model_dir, "oof_xgb_exp007.npy"), oof_xgb)
    np.save(os.path.join(model_dir, "oof_lgb_exp007.npy"), oof_lgb)

    print("\n  Top 20 Stage-1 Features by Gain (including New EXP-007 Features):")
    for rank, (fname, gain) in enumerate(sorted(s1_importances.items(), key=lambda x: x[1], reverse=True)[:20], 1):
        tag = " [NEW EXP-007]" if FEATURE_NAMES.index(fname) >= 40 else ""
        print(f"    {rank:>2}. {fname:<32} : {gain:>10.1f} gain{tag}")

    # 3. Train Stage-2 Group-Context Meta-Reranker (68 Features = 50 Pairwise + 18 Group Context)
    print("\n[Stage 3] Training Stage-2 Group-Context Meta-Reranker (68 Features, 5-Fold GroupKFold OOF)...")
    X_stage2 = build_stage2_context_features(X_val, oof_lgb, oof_xgb, group_lengths, blend_w_lgb=0.55)

    oof_s2_lgb = np.zeros(len(y_val), dtype=np.float32)
    oof_s2_xgb = np.zeros(len(y_val), dtype=np.float32)

    s2_lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 45,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "n_jobs": 12,
        "random_state": 42,
        "verbose": -1,
    }
    s2_xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_weight": 30.0,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "max_bin": 256,
        "seed": 42,
    }

    for fold in range(5):
        val_mask = (row_folds == fold)
        trn_mask = ~val_mask
        X_tr, y_tr = X_stage2[trn_mask], y_val[trn_mask]
        X_va, y_va = X_stage2[val_mask], y_val[val_mask]

        dtr_l = lgb.Dataset(X_tr, label=y_tr, feature_name=ALL_STAGE2_FEATURE_NAMES_EXP007)
        dva_l = lgb.Dataset(X_va, label=y_va, reference=dtr_l, feature_name=ALL_STAGE2_FEATURE_NAMES_EXP007)
        m_lgb = lgb.train(
            s2_lgb_params,
            dtr_l,
            num_boost_round=550,
            valid_sets=[dva_l],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        oof_s2_lgb[val_mask] = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)
        m_lgb.save_model(os.path.join(model_dir, f"stage2_lgb_exp007_fold{fold}.txt"), num_iteration=m_lgb.best_iteration)

        dtr_x = xgb.QuantileDMatrix(X_tr, label=y_tr, feature_names=ALL_STAGE2_FEATURE_NAMES_EXP007, max_bin=256)
        dva_x = xgb.QuantileDMatrix(X_va, label=y_va, ref=dtr_x, feature_names=ALL_STAGE2_FEATURE_NAMES_EXP007, max_bin=256)
        m_xgb = xgb.train(
            s2_xgb_params,
            dtr_x,
            num_boost_round=550,
            evals=[(dva_x, "val")],
            early_stopping_rounds=40,
            verbose_eval=False,
        )
        oof_s2_xgb[val_mask] = m_xgb.predict(dva_x, iteration_range=(0, m_xgb.best_iteration + 1))
        m_xgb.save_model(os.path.join(model_dir, f"stage2_xgb_exp007_fold{fold}.json"))

    oof_s1_blend = 0.55 * oof_lgb + 0.45 * oof_xgb
    oof_s2_ens = 0.50 * oof_s2_lgb + 0.50 * oof_s2_xgb
    oof_hier = 0.35 * oof_s1_blend + 0.65 * oof_s2_ens
    np.save(os.path.join(model_dir, "oof_stage2_exp007.npy"), oof_hier)

    # Also evaluate blending EXP-006 + EXP-007 OOF predictions
    oof_exp006 = np.load(os.path.join(model_dir, "oof_stage2_exp006.npy"))
    oof_super_ens = 0.80 * oof_hier + 0.20 * oof_exp006

    # 4. Evaluate Full-Pool OOF across India (7,812), US (12,188), and All 20,000 Val S1 Entities
    print("\n[Stage 4] Evaluating Full-Pool 5-Fold OOF Predictions (10,320,219 Candidate Pool)...")

    def build_groups(probs: np.ndarray) -> Dict[str, List[Tuple[str, float]]]:
        grps: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in val_s1}
        for sid, cid, p in zip(row_s1_ids, row_cand_ids, probs):
            grps[sid].append((cid, float(p)))
        for sid in grps:
            grps[sid].sort(key=lambda x: x[1], reverse=True)
        return grps

    grps_006 = build_groups(oof_exp006)
    grps_007 = build_groups(oof_hier)
    grps_ens = build_groups(oof_super_ens)

    m_006_base = evaluate_dual_threshold_rule(grps_006, val_gt, tau_top1=0.72, tau_sec=0.72, min_ratio=0.0, rescue_consensus_tau=0.68)

    best_cfg = None
    best_m_007 = None
    for tau_top1 in (0.66, 0.68, 0.70, 0.72, 0.74, 0.76):
        for tau_sec in (0.64, 0.66, 0.68, 0.70, 0.72, 0.74):
            if tau_sec > tau_top1:
                continue
            m = evaluate_dual_threshold_rule(
                grps_007, val_gt, tau_top1=tau_top1, tau_sec=tau_sec, min_ratio=0.0, rescue_consensus_tau=0.68
            )
            if best_m_007 is None or m["macro_f05"] > best_m_007["macro_f05"]:
                best_m_007 = m
                best_cfg = (tau_top1, tau_sec)

    t1, t2 = best_cfg
    m_007_dedup = evaluate_with_bipartite_dedup(
        grps_007, val_gt, tau_top1=t1, tau_sec=t2, min_ratio=0.0, rescue_consensus_tau=0.68, dedup_margin=0.0
    )
    m_ens_dedup = evaluate_with_bipartite_dedup(
        grps_ens, val_gt, tau_top1=t1, tau_sec=t2, min_ratio=0.0, rescue_consensus_tau=0.68, dedup_margin=0.0
    )

    india_ids = {sid for sid, r in val_s1.items() if r.country == "India"}
    us_ids = {sid for sid, r in val_s1.items() if r.country == "US"}
    m_in_007 = evaluate_with_bipartite_dedup(
        {sid: grps_007[sid] for sid in india_ids},
        {sid: val_gt[sid] for sid in india_ids},
        tau_top1=t1, tau_sec=t2, min_ratio=0.0, rescue_consensus_tau=0.68, dedup_margin=0.0,
    )
    m_us_007 = evaluate_with_bipartite_dedup(
        {sid: grps_007[sid] for sid in us_ids},
        {sid: val_gt[sid] for sid in us_ids},
        tau_top1=t1, tau_sec=t2, min_ratio=0.0, rescue_consensus_tau=0.68, dedup_margin=0.0,
    )

    # Evaluate Full-Population Bipartite Mutual Exclusion (where all 1.8M S1 entities compete for S2/S3 ownership,
    # matching the 100% test_source1.tsv evaluation regime)
    train_gt_path = os.path.join(PROJECT_ROOT, "dataset/train/train_ground_truth.tsv")
    ext_owned_cands: Set[str] = set()
    if os.path.isfile(train_gt_path):
        train_gt = load_ground_truth_tsv(train_gt_path)
        for sid, cset in train_gt.items():
            if sid not in val_gt:
                ext_owned_cands.update(cset)

    grps_fullpop = {
        sid: [(cid, p) for cid, p in pairs if cid not in ext_owned_cands or p >= 0.94]
        for sid, pairs in grps_007.items()
    }
    best_fullpop_m = None
    best_fp_cfg = (t1, t2)
    for fp_t1 in (0.60, 0.62, 0.64, 0.66, 0.68, 0.70):
        for fp_t2 in (0.56, 0.58, 0.60, 0.62, 0.64, 0.66):
            if fp_t2 > fp_t1:
                continue
            m_fp = evaluate_with_bipartite_dedup(
                grps_fullpop, val_gt, tau_top1=fp_t1, tau_sec=fp_t2, min_ratio=0.0, rescue_consensus_tau=0.62, dedup_margin=0.0
            )
            if best_fullpop_m is None or m_fp["macro_f05"] > best_fullpop_m["macro_f05"]:
                best_fullpop_m = m_fp
                best_fp_cfg = (fp_t1, fp_t2)

    print("-" * 106)
    print(
        f"{'Configuration / Subset':<32} | {'tau_1':>6} | {'tau_2':>6} | {'Macro F0.5':>10} | "
        f"{'Precision':>10} | {'Recall':>10} | {'Sing Acc':>9} | {'NonSing F0.5':>12}"
    )
    print("-" * 106)
    for label, tc1, tc2, m in [
        ("EXP-006 Baseline (40 Feats)", 0.72, 0.72, m_006_base),
        ("EXP-007b (50 Feats + GPU Ens)", t1, t2, best_m_007),
        ("EXP-007b + 20k Bipartite Dedup", t1, t2, m_007_dedup),
        ("EXP-007b+006 Super-Ensemble", t1, t2, m_ens_dedup),
        ("  -> India (7,812 S1)", t1, t2, m_in_007),
        ("  -> US (12,188 S1)", t1, t2, m_us_007),
        ("EXP-007b + Full-Pop Bipartite", best_fp_cfg[0], best_fp_cfg[1], best_fullpop_m),
    ]:
        print(
            f"{label:<32} | {tc1:>6.2f} | {tc2:>6.2f} | {m['macro_f05']:>10.4f} | "
            f"{m['macro_precision']:>10.4f} | {m['macro_recall']:>10.4f} | "
            f"{m['singleton_accuracy']*100:>8.2f}% | {m['non_singleton_f05']:>12.4f}"
        )
    print("-" * 106)
    print(f"Total EXP-007b GPU Pipeline Runtime: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
