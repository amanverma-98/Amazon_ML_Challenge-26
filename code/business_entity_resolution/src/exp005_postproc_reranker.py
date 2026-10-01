#!/usr/bin/env python3
"""
EXP-005: Entity-Level Post-Processing, Dual-Threshold Calibration & Stage-2 Group-Context Reranker.

Builds directly on the saved EXP-004 15-Lakh models (oof_xgb_exp004.npy and oof_lgb_exp004.npy):
  1. Evaluates Linear vs Log-Odds (Logit) Ensemble Blending between GPU XGBoost and LightGBM.
  2. Evaluates Entity-Level Dual-Thresholding (separating Singleton Gate tau_top1 from
     Secondary Match Threshold tau_sec + probability drop-off ratio) via 5-Fold OOF.
  3. Trains a Stage-2 Group-Context Meta-Reranker (30 Pairwise Features + 16 Entity-Group
     Probability Distribution & Cliff Features = 46 Features) using strict 5-Fold GroupKFold OOF,
     saving the Stage-2 models to output/models/ for fast Test Submission generation.
"""

import gc
import os
import pickle
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from code.business_entity_resolution.src.blocking import CountryPartitionedBlocker
from code.business_entity_resolution.src.data_loader import EntityRecord, load_ground_truth_tsv, load_source_tsv
from code.business_entity_resolution.src.features import FEATURE_NAMES, extract_features_for_s1_group
from code.business_entity_resolution.src.preprocessor import clean_address, clean_business_name, extract_name_variants
from utils.metrics import evaluate_predictions

_G_S1_RECORDS: Dict[str, EntityRecord] = {}
_G_CAND_POOL: Dict[str, EntityRecord] = {}
_G_GROUND_TRUTH: Dict[str, Set[str]] = {}
_G_TOKEN_DF: Dict[str, Dict[str, int]] = {}


def _val_worker(batch: List[Tuple[str, List[str]]]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Worker to extract 30 features for validation S1 entities and their top-35 candidates."""
    s1_order = []
    X_list = []
    y_list = []
    for s1_id, cand_ids in batch:
        if not cand_ids:
            continue
        s1_rec = _G_S1_RECORDS[s1_id]
        true_set = _G_GROUND_TRUTH.get(s1_id, set())
        c_recs = [_G_CAND_POOL[cid] for cid in cand_ids]
        c_df = _G_TOKEN_DF.get(s1_rec.country)

        feats = extract_features_for_s1_group(s1_rec, c_recs, country_token_df=c_df)
        labels = np.array([1.0 if cid in true_set else 0.0 for cid in cand_ids], dtype=np.float32)

        s1_order.append(s1_id)
        X_list.append(feats)
        y_list.append(labels)

    clean_business_name.cache_clear()
    extract_name_variants.cache_clear()
    clean_address.cache_clear()

    if not X_list:
        return [], np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return s1_order, np.vstack(X_list), np.concatenate(y_list)


STAGE2_CONTEXT_FEATURE_NAMES = [
    "p_lgb",
    "p_xgb",
    "p_blend",
    "logit_blend",
    "p_abs_diff",
    "p_min",
    "p_max",
    "s1_max_prob",
    "s1_second_max_prob",
    "prob_diff_from_s1_max",
    "prob_ratio_to_s1_max",
    "prob_rank_in_s1",
    "prob_diff_to_next_cand",
    "prob_diff_from_prev_cand",
    "s1_count_prob_gt_50",
    "s1_count_prob_gt_75",
    "s1_prob_sum",
    "s1_prob_std",
]

ALL_STAGE2_FEATURE_NAMES = list(FEATURE_NAMES) + STAGE2_CONTEXT_FEATURE_NAMES


def build_stage2_context_features(
    X_base: np.ndarray,
    p_lgb: np.ndarray,
    p_xgb: np.ndarray,
    group_lengths: List[int],
    blend_w_lgb: float = 0.60,
) -> np.ndarray:
    """
    Constructs the 48-dimensional Stage-2 feature matrix combining the 30 pairwise features
    with 18 entity-level probability context features computed within each S1 candidate group.
    Fast vectorized/slice implementation (~0.25s for 696,499 rows).
    """
    n_rows = len(p_lgb)
    ctx = np.zeros((n_rows, len(STAGE2_CONTEXT_FEATURE_NAMES)), dtype=np.float32)

    eps = 1e-6
    p_l = np.clip(p_lgb, eps, 1.0 - eps)
    p_x = np.clip(p_xgb, eps, 1.0 - eps)

    p_blend = (blend_w_lgb * p_l + (1.0 - blend_w_lgb) * p_x).astype(np.float32)
    logit_l = np.log(p_l / (1.0 - p_l))
    logit_x = np.log(p_x / (1.0 - p_x))
    logit_blend = (blend_w_lgb * logit_l + (1.0 - blend_w_lgb) * logit_x).astype(np.float32)

    ctx[:, 0] = p_l
    ctx[:, 1] = p_x
    ctx[:, 2] = p_blend
    ctx[:, 3] = logit_blend
    ctx[:, 4] = np.abs(p_l - p_x)
    ctx[:, 5] = np.minimum(p_l, p_x)
    ctx[:, 6] = np.maximum(p_l, p_x)

    offset = 0
    for length in group_lengths:
        if length <= 0:
            continue
        end = offset + length
        g_p = p_blend[offset:end]

        if length == 1:
            p0 = g_p[0]
            ctx[offset, 7] = p0
            ctx[offset, 8] = 0.0
            ctx[offset, 9] = 0.0
            ctx[offset, 10] = 1.0
            ctx[offset, 11] = 1.0
            ctx[offset, 12] = p0
            ctx[offset, 13] = 0.0
            ctx[offset, 14] = 1.0 if p0 >= 0.50 else 0.0
            ctx[offset, 15] = 1.0 if p0 >= 0.75 else 0.0
            ctx[offset, 16] = p0
            ctx[offset, 17] = 0.0
        else:
            order = np.argsort(-g_p)
            sorted_p = g_p[order]
            max_p = sorted_p[0]
            sec_p = sorted_p[1]

            ranks = np.empty(length, dtype=np.float32)
            ranks[order] = np.arange(1, length + 1, dtype=np.float32)

            # Next and previous candidate probability differences in sorted order
            diff_next_sorted = np.empty(length, dtype=np.float32)
            diff_next_sorted[:-1] = sorted_p[:-1] - sorted_p[1:]
            diff_next_sorted[-1] = sorted_p[-1]

            diff_prev_sorted = np.empty(length, dtype=np.float32)
            diff_prev_sorted[0] = 0.0
            diff_prev_sorted[1:] = sorted_p[:-1] - sorted_p[1:]

            diff_next = np.empty(length, dtype=np.float32)
            diff_prev = np.empty(length, dtype=np.float32)
            diff_next[order] = diff_next_sorted
            diff_prev[order] = diff_prev_sorted

            ctx[offset:end, 7] = max_p
            ctx[offset:end, 8] = sec_p
            ctx[offset:end, 9] = max_p - g_p
            ctx[offset:end, 10] = g_p / (max_p + 1e-6)
            ctx[offset:end, 11] = ranks
            ctx[offset:end, 12] = diff_next
            ctx[offset:end, 13] = diff_prev
            ctx[offset:end, 14] = float(np.sum(g_p >= 0.50))
            ctx[offset:end, 15] = float(np.sum(g_p >= 0.75))
            ctx[offset:end, 16] = float(np.sum(g_p))
            ctx[offset:end, 17] = float(np.std(g_p))

        offset = end

    return np.hstack([X_base, ctx])


def evaluate_dual_threshold_rule(
    s1_pred_groups: Dict[str, List[Tuple[str, float]]],
    val_gt: Dict[str, Set[str]],
    tau_top1: float,
    tau_sec: float,
    min_ratio: float = 0.0,
    rescue_consensus_tau: float = 1.0,
) -> Dict[str, float]:
    """
    Applies entity-aware decision rule:
      - An S1 entity is predicted as Non-Singleton iff:
          (a) its #1 candidate has P_1 >= tau_top1, OR
          (b) its top 2 candidates BOTH have P_1 >= rescue_consensus_tau and P_2 >= rescue_consensus_tau
              (multi-record cluster consensus).
      - Once an entity is Non-Singleton, any candidate k with P_k >= tau_sec and (P_k / P_1) >= min_ratio
        is included in the predicted match set.
    """
    preds: Dict[str, Set[str]] = {}
    for s1_id, pairs in s1_pred_groups.items():
        if not pairs:
            preds[s1_id] = set()
            continue
        # pairs is pre-sorted descending by probability
        p1 = pairs[0][1]
        p2 = pairs[1][1] if len(pairs) > 1 else 0.0

        is_non_singleton = (p1 >= tau_top1) or (p1 >= rescue_consensus_tau and p2 >= rescue_consensus_tau)
        if not is_non_singleton:
            preds[s1_id] = set()
            continue

        matched = {pairs[0][0]}
        for cid, prob in pairs[1:]:
            if prob >= tau_sec and (prob / (p1 + 1e-6)) >= min_ratio:
                matched.add(cid)
            else:
                break
        preds[s1_id] = matched

    return evaluate_predictions(val_gt, preds)


def main():
    global _G_S1_RECORDS, _G_CAND_POOL, _G_GROUND_TRUTH, _G_TOKEN_DF

    val_dir = os.path.join(PROJECT_ROOT, "dataset/val")
    model_dir = os.path.join(PROJECT_ROOT, "output/models")
    os.makedirs(model_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 92)
    print("🚀 RUNNING EXP-005: ENTITY-LEVEL POST-PROCESSING & STAGE-2 GROUP-CONTEXT RERANKER")
    print("=" * 92, flush=True)

    # 1. Load Validation Benchmark
    val_s1 = load_source_tsv(os.path.join(val_dir, "val_source1.tsv"))
    val_gt = load_ground_truth_tsv(os.path.join(val_dir, "val_ground_truth.tsv"))

    cache_path = os.path.join(model_dir, "val_matrix_exp004.npz")
    meta_cache_path = os.path.join(model_dir, "val_meta_exp004.pkl")

    if os.path.isfile(cache_path) and os.path.isfile(meta_cache_path):
        print("\n[1/4] Loading cached Validation Feature Matrix & Candidate Mapping...")
        data = np.load(cache_path)
        X_val = data["X_val"]
        y_val = data["y_val"]
        with open(meta_cache_path, "rb") as f:
            meta = pickle.load(f)
        row_s1_ids = meta["row_s1_ids"]
        row_cand_ids = meta["row_cand_ids"]
        group_lengths = meta["group_lengths"]
        print(f"  Loaded X_val: {X_val.shape} ({time.time() - t_start:.2f}s)")
    else:
        print("\n[1/4] Building & Caching Validation Feature Matrix (696,499 pairs x 30 features)...")
        val_s2 = load_source_tsv(os.path.join(val_dir, "val_source2.tsv"))
        val_s3 = load_source_tsv(os.path.join(val_dir, "val_source3.tsv"))
        val_pool: Dict[str, EntityRecord] = {}
        val_pool.update(val_s2)
        val_pool.update(val_s3)

        blocker = CountryPartitionedBlocker(max_postings_per_key=2500, top_k=35)
        blocker.fit(val_s2, val_s3)
        val_cands_map = blocker.generate_candidates(val_s1)
        token_df = blocker.token_df
        blocker.index.clear()
        blocker.key_weights.clear()
        del val_s2, val_s3
        gc.collect()

        _G_S1_RECORDS = val_s1
        _G_CAND_POOL = val_pool
        _G_GROUND_TRUTH = val_gt
        _G_TOKEN_DF = token_df

        val_items = list(val_cands_map.items())
        batch_size = 1000
        val_batches = [val_items[i:i + batch_size] for i in range(0, len(val_items), batch_size)]

        val_X_parts = []
        val_y_parts = []
        gc.freeze()
        with Pool(processes=4) as pool:
            for _, X_b, y_b in pool.imap(_val_worker, val_batches):
                val_X_parts.append(X_b)
                val_y_parts.append(y_b)
        gc.unfreeze()

        X_val = np.vstack(val_X_parts)
        y_val = np.concatenate(val_y_parts)
        del val_X_parts, val_y_parts, val_pool
        _G_S1_RECORDS = {}
        _G_CAND_POOL = {}
        _G_GROUND_TRUTH = {}
        gc.collect()

        row_s1_ids = []
        row_cand_ids = []
        group_lengths = []
        for s1_id, cand_ids in val_items:
            if not cand_ids:
                continue
            group_lengths.append(len(cand_ids))
            for cid in cand_ids:
                row_s1_ids.append(s1_id)
                row_cand_ids.append(cid)

        np.savez(cache_path, X_val=X_val, y_val=y_val)
        with open(meta_cache_path, "wb") as f:
            pickle.dump(
                {
                    "row_s1_ids": row_s1_ids,
                    "row_cand_ids": row_cand_ids,
                    "group_lengths": group_lengths,
                },
                f,
            )
        print(f"  Cached X_val ({X_val.shape}) to {cache_path} ({time.time() - t_start:.2f}s)")

    # Load saved Stage-1 OOF predictions from EXP-004
    oof_xgb = np.load(os.path.join(model_dir, "oof_xgb_exp004.npy"))
    oof_lgb = np.load(os.path.join(model_dir, "oof_lgb_exp004.npy"))

    # 2. Evaluate Linear vs Log-Odds (Logit) Blending Weights
    print("\n[2/4] Optimizing Ensemble Blend Weights (Linear vs Log-Odds Probability Calibration)...")
    print("-" * 92)
    print(f"{'Blend Method':<28} | {'Thresh':>6} | {'Macro F0.5':>10} | {'Precision':>10} | {'Recall':>10} | {'Sing Acc':>10} | {'NonSing F0.5':>12}")
    print("-" * 92)

    eps = 1e-6
    p_l = np.clip(oof_lgb, eps, 1.0 - eps)
    p_x = np.clip(oof_xgb, eps, 1.0 - eps)
    logit_l = np.log(p_l / (1.0 - p_l))
    logit_x = np.log(p_x / (1.0 - p_x))

    best_blend_probs = None
    best_blend_score = -1.0
    best_blend_name = ""

    blend_candidates = [
        ("XGBoost Only (0.0/1.0)", oof_xgb),
        ("LightGBM Only (1.0/0.0)", oof_lgb),
        ("Linear 50% LGB + 50% XGB", 0.50 * oof_lgb + 0.50 * oof_xgb),
        ("Linear 60% LGB + 40% XGB", 0.60 * oof_lgb + 0.40 * oof_xgb),
        ("Linear 65% LGB + 35% XGB", 0.65 * oof_lgb + 0.35 * oof_xgb),
        ("Log-Odds 50% LGB + 50% XGB", 1.0 / (1.0 + np.exp(-(0.50 * logit_l + 0.50 * logit_x)))),
        ("Log-Odds 60% LGB + 40% XGB", 1.0 / (1.0 + np.exp(-(0.60 * logit_l + 0.40 * logit_x)))),
    ]

    for b_name, b_probs in blend_candidates:
        s1_groups: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in val_s1}
        for sid, cid, prob in zip(row_s1_ids, row_cand_ids, b_probs):
            s1_groups[sid].append((cid, float(prob)))

        best_t = 0.78
        best_m = None
        for tau in (0.74, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81):
            preds = {sid: {cid for cid, p in pairs if p >= tau} for sid, pairs in s1_groups.items()}
            m = evaluate_predictions(val_gt, preds)
            if best_m is None or m["macro_f05"] > best_m["macro_f05"]:
                best_m = m
                best_t = tau

        print(
            f"{b_name:<28} | {best_t:>6.2f} | {best_m['macro_f05']:>10.4f} | "
            f"{best_m['macro_precision']:>10.4f} | {best_m['macro_recall']:>10.4f} | "
            f"{best_m['singleton_accuracy']*100:>9.2f}% | {best_m['non_singleton_f05']:>12.4f}"
        )
        if best_m["macro_f05"] > best_blend_score:
            best_blend_score = best_m["macro_f05"]
            best_blend_probs = b_probs
            best_blend_name = b_name

    # 3. Train Stage-2 Group-Context Meta-Reranker (5-Fold GroupKFold OOF)
    print("\n[3/4] Training Stage-2 Group-Context Meta-Reranker (48 Features = 30 Pairwise + 18 Group Prob Context)...")
    t_s2 = time.time()
    X_stage2 = build_stage2_context_features(X_val, oof_lgb, oof_xgb, group_lengths, blend_w_lgb=0.60)
    print(f"  Constructed Stage-2 Feature Matrix: {X_stage2.shape} in {time.time() - t_s2:.2f}s")

    unique_val_s1 = list(val_s1.keys())
    s1_to_fold = {s1_id: (idx % 5) for idx, s1_id in enumerate(unique_val_s1)}
    row_folds = np.array([s1_to_fold[sid] for sid in row_s1_ids], dtype=np.int32)

    oof_s2_lgb = np.zeros(len(y_val), dtype=np.float32)
    oof_s2_xgb = np.zeros(len(y_val), dtype=np.float32)

    s2_lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 60,
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
        "min_child_weight": 40.0,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "max_bin": 256,
        "seed": 42,
    }

    s2_importances = {f: 0.0 for f in ALL_STAGE2_FEATURE_NAMES}

    for fold in range(5):
        val_mask = (row_folds == fold)
        trn_mask = ~val_mask

        X_va, y_va = X_stage2[val_mask], y_val[val_mask]
        lgb_s2_path = os.path.join(model_dir, f"stage2_lgb_exp005_fold{fold}.txt")
        xgb_s2_path = os.path.join(model_dir, f"stage2_xgb_exp005_fold{fold}.json")

        if os.path.isfile(lgb_s2_path) and os.path.isfile(xgb_s2_path):
            m_lgb = lgb.Booster(model_file=lgb_s2_path)
            oof_s2_lgb[val_mask] = m_lgb.predict(X_va)
            gains = m_lgb.feature_importance(importance_type="gain")
            for fname, gval in zip(ALL_STAGE2_FEATURE_NAMES, gains):
                s2_importances[fname] += float(gval) / 5.0

            m_xgb = xgb.Booster()
            m_xgb.load_model(xgb_s2_path)
            dva_xgb = xgb.DMatrix(X_va, label=y_va, feature_names=ALL_STAGE2_FEATURE_NAMES)
            oof_s2_xgb[val_mask] = m_xgb.predict(dva_xgb)
            continue

        X_tr, y_tr = X_stage2[trn_mask], y_val[trn_mask]

        # Stage-2 LightGBM
        dtr_lgb = lgb.Dataset(X_tr, label=y_tr, feature_name=ALL_STAGE2_FEATURE_NAMES)
        dva_lgb = lgb.Dataset(X_va, label=y_va, reference=dtr_lgb, feature_name=ALL_STAGE2_FEATURE_NAMES)
        m_lgb = lgb.train(
            s2_lgb_params,
            dtr_lgb,
            num_boost_round=500,
            valid_sets=[dva_lgb],
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
        )
        oof_s2_lgb[val_mask] = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)
        m_lgb.save_model(lgb_s2_path, num_iteration=m_lgb.best_iteration)

        gains = m_lgb.feature_importance(importance_type="gain")
        for fname, gval in zip(ALL_STAGE2_FEATURE_NAMES, gains):
            s2_importances[fname] += float(gval) / 5.0

        # Stage-2 GPU XGBoost
        dtr_xgb = xgb.QuantileDMatrix(X_tr, label=y_tr, feature_names=ALL_STAGE2_FEATURE_NAMES, max_bin=256)
        dva_xgb = xgb.QuantileDMatrix(X_va, label=y_va, ref=dtr_xgb, feature_names=ALL_STAGE2_FEATURE_NAMES, max_bin=256)
        m_xgb = xgb.train(
            s2_xgb_params,
            dtr_xgb,
            num_boost_round=500,
            evals=[(dva_xgb, "val")],
            early_stopping_rounds=35,
            verbose_eval=False,
        )
        oof_s2_xgb[val_mask] = m_xgb.predict(dva_xgb, iteration_range=(0, m_xgb.best_iteration + 1))
        m_xgb.save_model(xgb_s2_path)

    oof_s2_ens = 0.50 * oof_s2_lgb + 0.50 * oof_s2_xgb
    # Also test blending Stage-1 + Stage-2
    oof_hier_blend = 0.40 * (0.60 * oof_lgb + 0.40 * oof_xgb) + 0.60 * oof_s2_ens

    np.save(os.path.join(model_dir, "oof_stage2_exp005.npy"), oof_hier_blend)
    print(f"  Stage-2 5-Fold GroupKFold OOF Training completed in {time.time() - t_s2:.2f}s!")

    print("\n  Top 12 Features in Stage-2 Group-Context Reranker:")
    for rank, (fname, gain) in enumerate(sorted(s2_importances.items(), key=lambda x: x[1], reverse=True)[:12], 1):
        print(f"    {rank:>2}. {fname:<28} : {gain:>10.1f} gain")

    # Evaluate Stage-2 OOF across Flat Thresholds
    print("\n📊 Stage-2 Group-Context Reranker Flat Threshold Sweep:")
    print("-" * 92)
    print(f"{'Model Variant':<32} | {'Thresh':>6} | {'Macro F0.5':>10} | {'Precision':>10} | {'Recall':>10} | {'Sing Acc':>10} | {'NonSing F0.5':>12}")
    print("-" * 92)

    for v_name, v_probs in [
        ("Stage-2 LightGBM OOF", oof_s2_lgb),
        ("Stage-2 GPU XGBoost OOF", oof_s2_xgb),
        ("Stage-2 Ensemble (LGB+XGB)", oof_s2_ens),
        ("Hierarchical Stage-1 + Stage-2", oof_hier_blend),
    ]:
        s1_groups = {sid: [] for sid in val_s1}
        for sid, cid, prob in zip(row_s1_ids, row_cand_ids, v_probs):
            s1_groups[sid].append((cid, float(prob)))

        best_t = 0.78
        best_m = None
        for tau in (0.72, 0.74, 0.76, 0.78, 0.79, 0.80, 0.81, 0.82):
            preds = {sid: {cid for cid, p in pairs if p >= tau} for sid, pairs in s1_groups.items()}
            m = evaluate_predictions(val_gt, preds)
            if best_m is None or m["macro_f05"] > best_m["macro_f05"]:
                best_m = m
                best_t = tau
        print(
            f"{v_name:<32} | {best_t:>6.2f} | {best_m['macro_f05']:>10.4f} | "
            f"{best_m['macro_precision']:>10.4f} | {best_m['macro_recall']:>10.4f} | "
            f"{best_m['singleton_accuracy']*100:>9.2f}% | {best_m['non_singleton_f05']:>12.4f}"
        )

    # 4. Optimize Entity-Level Dual-Threshold Rule on Hierarchical Stage-1 + Stage-2 OOF
    print("\n[4/4] Optimizing Entity-Level Dual-Threshold Rule (Singleton Gate tau_top1 vs Secondary tau_sec)...")
    s1_sorted_groups: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in val_s1}
    for sid, cid, prob in zip(row_s1_ids, row_cand_ids, oof_hier_blend):
        s1_sorted_groups[sid].append((cid, float(prob)))
    for sid in s1_sorted_groups:
        s1_sorted_groups[sid].sort(key=lambda x: x[1], reverse=True)

    print("-" * 98)
    print(
        f"{'tau_top1':>8} | {'tau_sec':>8} | {'min_ratio':>9} | {'Macro F0.5':>10} | "
        f"{'Precision':>10} | {'Recall':>10} | {'Sing Acc':>10} | {'NonSing F0.5':>12}"
    )
    print("-" * 98)

    best_dual_cfg = None
    best_dual_m = None

    for tau_top1 in (0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.80):
        for tau_sec in (0.68, 0.70, 0.72, 0.74, 0.75, 0.76, 0.78):
            if tau_sec > tau_top1:
                continue
            for min_ratio in (0.0, 0.80, 0.85):
                m = evaluate_dual_threshold_rule(
                    s1_sorted_groups,
                    val_gt,
                    tau_top1=tau_top1,
                    tau_sec=tau_sec,
                    min_ratio=min_ratio,
                    rescue_consensus_tau=0.73,
                )
                if best_dual_m is None or m["macro_f05"] > best_dual_m["macro_f05"]:
                    best_dual_m = m
                    best_dual_cfg = (tau_top1, tau_sec, min_ratio)

    # Print representative rows and the winning configuration
    for tau_top1, tau_sec, min_ratio in [
        (0.73, 0.72, 0.00),
        (0.74, 0.72, 0.00),
        (0.74, 0.74, 0.00),
        (0.75, 0.72, 0.80),
        (0.75, 0.74, 0.00),
        (0.76, 0.74, 0.00),
        (0.76, 0.76, 0.00),
        best_dual_cfg,
    ]:
        m = evaluate_dual_threshold_rule(
            s1_sorted_groups,
            val_gt,
            tau_top1=tau_top1,
            tau_sec=tau_sec,
            min_ratio=min_ratio,
            rescue_consensus_tau=0.73,
        )
        print(
            f"{tau_top1:>8.2f} | {tau_sec:>8.2f} | {min_ratio:>9.2f} | {m['macro_f05']:>10.4f} | "
            f"{m['macro_precision']:>10.4f} | {m['macro_recall']:>10.4f} | "
            f"{m['singleton_accuracy']*100:>9.2f}% | {m['non_singleton_f05']:>12.4f}"
        )

    print("-" * 98)
    print(
        f"🏆 EXP-005 Best Configuration (tau_top1={best_dual_cfg[0]:.2f}, tau_sec={best_dual_cfg[1]:.2f}, "
        f"min_ratio={best_dual_cfg[2]:.2f}):\n"
        f"   --> Macro F_0.5 = {best_dual_m['macro_f05']:.4f} "
        f"(Precision: {best_dual_m['macro_precision']:.4f}, Recall: {best_dual_m['macro_recall']:.4f}, "
        f"Singleton Acc: {best_dual_m['singleton_accuracy']*100:.2f}%, "
        f"Non-Singleton F_0.5: {best_dual_m['non_singleton_f05']:.4f})"
    )
    print(f"\nTotal EXP-005 Runtime: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
