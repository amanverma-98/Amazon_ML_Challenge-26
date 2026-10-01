"""
Pairwise and Group-Relative Feature Engineering module for Business Entity Resolution (EXP-003).
Extracts 26 domain-specific, country-agnostic numerical features for each (S1, Candidate) pair.
"""

import math
import re
from typing import Dict, List, Optional
import numpy as np
from rapidfuzz import distance, fuzz

from collections import Counter
from code.business_entity_resolution.src.blocking import ADDR_STOP_TOKENS, CITY_TOKENS, GENERIC_CORP_WORDS, PURE_STOP_WORDS, STATE_TOKENS
from code.business_entity_resolution.src.data_loader import EntityRecord
from code.business_entity_resolution.src.matcher import extract_numbers, is_single_coined_alias
from code.business_entity_resolution.src.preprocessor import (
    INDIC_RANGE_RE,
    _SKEL_REPEAT_RE,
    clean_address,
    clean_business_name,
    extract_name_variants,
    phonetic_skeleton_word,
)

MULTI_DIGIT_RE = re.compile(r"(?<!\d)\d{2,}(?!\d)")
ALL_DIGIT_RE = re.compile(r"\d+")
HYPH_NUM_RE = re.compile(r"(?<!\d)(\d+)\s*[-/]\s*(\d+)(?!\d)")

FULL_NAME_STOP_WORDS = PURE_STOP_WORDS | {"india", "usa", "us", "france"}
LOC_TOKENS = (CITY_TOKENS | STATE_TOKENS) - {"pradesh", "nadu"}
NON_STREET_ADDR_TOKENS = ADDR_STOP_TOKENS | LOC_TOKENS
_SYN_GENERIC_SUB_WORDS = {
    "service", "services", "partners", "partner", "center", "centre",
    "group", "solutions", "enterprises", "enterprise",
}
_SOFT_CORP_DIFF_TOKENS = {"group", "company", "co"}
_CROSS_SCRIPT_ACRONYM_MAP = {"aiti": "it", "eses": "ss", "eiai": "ai"}

BRANCH_OR_ROMAN_TOKENS = {
    "ii", "iii", "iv", "vi", "vii", "viii", "ix",
    "2", "3", "4", "5",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "1st", "2nd", "3rd", "4th", "5th",
    "north", "south", "east", "west",
}

_MED_CRED_TOKENS = {"md", "dmd", "dds", "dpm", "do", "od", "dc", "pa", "np", "rn", "cpa", "phd"}
_CORRUPTED_MED_TOKENS = {"dum", "rpm", "dpmb", "gmd", "hmd", "ddmd"}

FEATURE_NAMES = [
    "name_sort_sim",
    "name_set_sim",
    "name_jw_sim",
    "name_lev_ratio",
    "name_spaceless_sim",
    "full_name_sort_sim",
    "name_token_jaccard",
    "name_prefix_match",
    "name_len_ratio",
    "s1_name_word_count",
    "s1_name_min_df",
    "shared_name_idf_sum",
    "addr_is_missing",
    "addr_sort_sim",
    "addr_set_sim",
    "addr_jw_sim",
    "street_rare_word_jaccard",
    "shared_multi_digit_nums",
    "multi_digit_num_jaccard",
    "primary_street_num_match",
    "any_multi_digit_conflict",
    "is_coined_alias",
    "heuristic_score",
    "cand_rank_by_heuristic",
    "diff_from_s1_max_heuristic",
    "diff_from_s1_max_addr_sim",
    "min_unmatched_name_toks",
    "max_unmatched_name_toks",
    "roman_or_branch_conflict",
    "primary_street_num_sim",
    "full_name_token_jaccard",
    "full_min_unmatched_name_toks",
    "full_max_unmatched_name_toks",
    "all_num_jaccard",
    "both_unmatched_all_nums",
    "unmatched_all_nums_count",
    "unmatched_num_max_sim",
    "city_or_state_conflict",
    "legal_type_conflict",
    "first_word_sim",
    "pure_street_sort_sim",
    "pure_street_set_sim",
    "min_pure_street_words",
    "shared_3plus_digit_nums",
    "unmatched_2plus_digit_nums_cnt",
    "s1_same_name_match_addr_cnt",
    "s1_same_name_diff_addr_cnt",
    "cand_addr_cluster_size",
    "sim_to_top_addr_cand_addr",
    "cred_or_short_code_conflict",
]

_PVT_TOKENS = {"private", "pvt", "pvtltd", "praivet", "piraivet", "praibhet", "praivat", "praivarr"}
_LLP_TOKENS = {"llp", "elelpi", "ailaailapi", "ailailpi", "partnership"}
_PUB_TOKENS = {"public", "plc"}
_LLC_TOKENS = {"llc", "pllc"}
_INC_TOKENS = {"inc", "incorporated", "corp", "corporation"}
_LTD_TOKENS = {"ltd", "limited", "limitet", "limitad", "limirrad", "limatid", "limatida"}
_FR_SARL_TOKENS = {"sarl"}
_FR_SAS_TOKENS = {"sas", "sasu"}
_FR_EURL_TOKENS = {"eurl"}
_FR_SCI_TOKENS = {"sci"}
_FR_SA_TOKENS = {"sa"}
_FR_SNC_TOKENS = {"snc"}


def _extract_legal_class(full_clean: str, stripped_clean: str) -> int:
    """
    Returns legal structure class:
      1 = LLP, 2 = PVT, 3 = PUB, 4 = LLC, 5 = INC/CORP, 6 = LTD (generic),
      7 = SARL, 8 = SAS/SASU, 9 = EURL, 10 = SCI, 11 = SA, 12 = SNC, 0 = NONE
    """
    toks = set(full_clean.split())
    if toks & _LLP_TOKENS:
        return 1
    if toks & _PUB_TOKENS:
        return 3
    if (toks & _PVT_TOKENS) or stripped_clean.endswith("private"):
        return 2
    if toks & _LLC_TOKENS:
        return 4
    if toks & _INC_TOKENS:
        return 5
    if toks & _LTD_TOKENS:
        return 6
    if (toks & _FR_EURL_TOKENS) or ("unipersonnelle" in toks and "responsabilite" in toks):
        return 9
    if (toks & _FR_SARL_TOKENS) or ("responsabilite" in toks and "limitee" in toks):
        return 7
    if (toks & _FR_SAS_TOKENS) or ("actions" in toks and "simplifiee" in toks):
        return 8
    if (toks & _FR_SCI_TOKENS) or ("civile" in toks and "immobiliere" in toks):
        return 10
    if (toks & _FR_SNC_TOKENS) or ("nom" in toks and "collectif" in toks):
        return 12
    if (toks & _FR_SA_TOKENS) or ("societe" in toks and "anonyme" in toks):
        return 11
    return 0


def _is_legal_conflict(l1: int, l2: int, full_toks1: Optional[set] = None, full_toks2: Optional[set] = None) -> float:
    """
    Returns 3-level legal structure conflict:
      1.0 = Hard incompatible legal entity types (e.g. LLP vs PVT/LTD, PUB vs PVT, LLC vs INC/CORP, SARL vs SAS).
      0.5 = Soft legal suffix difference (e.g. NONE vs PVT/LTD, PVT vs generic LTD, or 'group'/'company' added).
      0.0 = Exact legal structure match.
    """
    if l1 != l2:
        if l1 == 0 or l2 == 0:
            return 0.5
        pair = (l1, l2) if l1 < l2 else (l2, l1)
        if pair in ((2, 6), (3, 6)):
            return 0.5
        return 1.0
    if full_toks1 is not None and full_toks2 is not None:
        if (full_toks1 ^ full_toks2) & _SOFT_CORP_DIFF_TOKENS:
            return 0.5
    return 0.0


def _align_cross_script_text(target_text: str, ref_skel_map: Dict[str, str], ref_full_skel: str = "", ref_full_text: str = "", ref_tokens: Optional[set] = None) -> str:
    """
    Aligns transliterated Indic tokens in target_text to their canonical Latin counterparts
    in ref_skel_map when their phonetic consonant skeletons match.
    """
    if not target_text or not ref_skel_map:
        return target_text
    tokens = target_text.split()
    skels = [phonetic_skeleton_word(t) if not t.isdigit() else t for t in tokens]
    if ref_full_skel and len(ref_full_skel) >= 4:
        full_sk = _SKEL_REPEAT_RE.sub(r"\1", "".join(s for s in skels if s))
        if full_sk == ref_full_skel:
            return ref_full_text

    out = []
    for tok, sk in zip(tokens, skels):
        if ref_tokens and tok in _CROSS_SCRIPT_ACRONYM_MAP and _CROSS_SCRIPT_ACRONYM_MAP[tok] in ref_tokens:
            out.append(_CROSS_SCRIPT_ACRONYM_MAP[tok])
        elif len(sk) >= 2 and sk in ref_skel_map:
            out.append(ref_skel_map[sk])
        elif len(sk) >= 3:
            best_w = tok
            for ref_sk, ref_w in ref_skel_map.items():
                if len(ref_sk) >= 3 and sk[0] == ref_sk[0]:
                    thresh = 74.0 if (len(sk) >= 4 and len(ref_sk) >= 4) else 82.0
                    if fuzz.ratio(sk, ref_sk) >= thresh:
                        best_w = ref_w
                        break
            out.append(best_w)
        else:
            out.append(tok)
    return " ".join(out)


def _count_unmatched_tokens(src_tokens: set, dst_tokens: set) -> int:
    """
    Counts how many tokens in src_tokens have NO exact or valid typo match in dst_tokens.
    Requires same starting character and length >= 5 for fuzzy typo matching so short acronyms
    ('ved' vs 'jved', 'qc' vs 'kqc') and distinct words sharing a 4-char prefix ('sarastech' vs 'sarastetis')
    are properly flagged as unmatched.
    """
    if not src_tokens:
        return 0
    if not dst_tokens:
        return len(src_tokens)
    unmatched = 0
    for t1 in src_tokens:
        if t1 in dst_tokens:
            continue
        matched = False
        l1 = len(t1)
        for t2 in dst_tokens:
            l2 = len(t2)
            if t1[0] == t2[0] and min(l1, l2) >= 5 and fuzz.ratio(t1, t2) >= 83.0:
                matched = True
                break
        if not matched:
            unmatched += 1
    return unmatched


def extract_multi_digit_numbers(addr: str) -> List[str]:
    """Extracts ordered multi-digit numbers (>=2 digits after stripping leading zeros)."""
    res = []
    for m in MULTI_DIGIT_RE.findall(addr):
        norm = m.lstrip("0")
        if len(norm) >= 2:
            res.append(norm)
    return res


def extract_features_for_s1_group(
    s1_rec: EntityRecord,
    cand_records: List[EntityRecord],
    country_token_df: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Computes the (len(cand_records), 30) float32 feature matrix for a single S1 entity
    and all of its blocked candidate records, including group-relative ranking features.
    """
    n_cands = len(cand_records)
    if n_cands == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    feats = np.zeros((n_cands, len(FEATURE_NAMES)), dtype=np.float32)

    # Pre-extract S1 properties once
    full_n1, strip_n1 = clean_business_name(s1_rec.business_name)
    vars1 = extract_name_variants(s1_rec.business_name)
    s1_has_indic_name = bool(INDIC_RANGE_RE.search(s1_rec.business_name))
    s1_has_indic_addr = bool(INDIC_RANGE_RE.search(s1_rec.business_address))

    s1_name_skel_map: Dict[str, str] = {}
    s1_skel_list: List[str] = []
    for t in strip_n1.split():
        if not t.isdigit():
            sk = phonetic_skeleton_word(t)
            if len(sk) >= 2:
                s1_name_skel_map[sk] = t
                if t not in PURE_STOP_WORDS:
                    s1_skel_list.append(sk)
    s1_full_skel = _SKEL_REPEAT_RE.sub(r"\1", "".join(s1_skel_list)) if s1_skel_list else ""

    s1_sig_tokens = [
        t for t in strip_n1.split()
        if len(t) >= 2 and t not in GENERIC_CORP_WORDS
    ]
    s1_token_set = set(s1_sig_tokens)
    s1_full_list = [
        t for t in strip_n1.split()
        if len(t) >= 2 and t not in FULL_NAME_STOP_WORDS
    ]
    s1_full_tokens = set(s1_full_list)
    s1_nonstop_list = [t for t in strip_n1.split() if t not in PURE_STOP_WORDS]
    s1_inits = {
        s for s in (
            "".join(t[0] for t in s1_sig_tokens),
            "".join(t[0] for t in s1_full_list),
            "".join(t[0] for t in s1_nonstop_list),
        )
        if 2 <= len(s) <= 4
    }
    s1_word_count = float(len(s1_sig_tokens))
    s1_full_raw_toks = set(full_n1.split())
    s1_branch_tokens = s1_full_raw_toks & BRANCH_OR_ROMAN_TOKENS

    if country_token_df is not None and s1_sig_tokens:
        s1_min_df = float(min(country_token_df.get(t, 0) for t in s1_sig_tokens))
    else:
        s1_min_df = 500.0

    # Also pre-extract vars1[0] tokens for missing-address heuristic check
    v1_0_sig_tokens = [
        t for t in vars1[0].split()
        if len(t) >= 2 and t not in GENERIC_CORP_WORDS
    ]
    v1_0_min_df = (
        min(country_token_df.get(t, 0) for t in v1_0_sig_tokens)
        if (country_token_df is not None and v1_0_sig_tokens)
        else 0
    )

    addr1 = clean_address(s1_rec.business_address, s1_rec.country)
    s1_addr_skel_map = {
        phonetic_skeleton_word(t): t
        for t in addr1.split()
        if not t.isdigit() and len(t) >= 3 and len(phonetic_skeleton_word(t)) >= 3
    }
    s1_nums = extract_multi_digit_numbers(addr1)
    s1_num_set = set(s1_nums)
    s1_all_nums = extract_numbers(addr1)
    s1_all_num_list = [m.lstrip("0") or "0" for m in ALL_DIGIT_RE.findall(addr1)]
    s1_all_num_counter = Counter(s1_all_num_list)
    s1_hyph_joins = {
        (m1.lstrip("0") or "0") + m2
        for m1, m2 in HYPH_NUM_RE.findall(s1_rec.business_address)
    } | {
        (m1.lstrip("0") or "0") + (m2.lstrip("0") or "0")
        for m1, m2 in HYPH_NUM_RE.findall(s1_rec.business_address)
    }
    s1_loc_tokens = set(addr1.split()) & LOC_TOKENS
    if "telangana" in s1_loc_tokens or "andhra" in s1_loc_tokens:
        s1_loc_tokens = s1_loc_tokens | {"telangana", "andhra"}
    s1_rare_addr_words = {
        t for t in addr1.split()
        if len(t) >= 3 and not t.isdigit() and t not in ADDR_STOP_TOKENS
    }
    s1_pure_street_list = [
        t for t in addr1.split()
        if len(t) >= 3 and t.isalpha() and t not in NON_STREET_ADDR_TOKENS
    ]
    s1_pure_street_str = " ".join(s1_pure_street_list)
    s1_pure_street_cnt = len(s1_pure_street_list)
    s1_3plus_nums = {n for n in s1_num_set if len(n) >= 3}

    s1_legal_class = _extract_legal_class(full_n1, strip_n1)
    s1_first_word = next((t for t in strip_n1.split() if len(t) >= 2 and t not in FULL_NAME_STOP_WORDS), "")
    s1_coined = is_single_coined_alias(s1_rec.business_name, vars1[0])

    cand_clean_addrs: List[str] = [""] * n_cands

    for idx, cand_rec in enumerate(cand_records):
        full_n2, strip_n2 = clean_business_name(cand_rec.business_name)
        cand_legal_class = _extract_legal_class(full_n2, strip_n2)
        cand_full_raw_toks = set(full_n2.split())
        legal_conflict = _is_legal_conflict(s1_legal_class, cand_legal_class, s1_full_raw_toks, cand_full_raw_toks)
        vars2 = extract_name_variants(cand_rec.business_name)

        cand_has_indic_name = bool(INDIC_RANGE_RE.search(cand_rec.business_name))
        if s1_has_indic_name != cand_has_indic_name:
            strip_n2 = _align_cross_script_text(strip_n2, s1_name_skel_map, s1_full_skel, strip_n1, s1_full_tokens)
            full_n2 = _align_cross_script_text(full_n2, s1_name_skel_map, ref_tokens=s1_full_tokens)
            vars2 = tuple(_align_cross_script_text(v2, s1_name_skel_map, s1_full_skel, vars1[0], s1_full_tokens) for v2 in vars2)

        best_sort = 0.0
        best_set = 0.0
        best_jw = 0.0
        best_lev = 0.0
        best_spaceless = 0.0
        best_len_ratio = 0.0
        best_name_score = 0.0
        best_strict_name = 0.0

        for v1 in vars1:
            s_v1 = v1.replace(" ", "")
            for v2 in vars2:
                s_v2 = v2.replace(" ", "")
                n_sort = fuzz.token_sort_ratio(v1, v2)
                n_set = fuzz.token_set_ratio(v1, v2)
                n_jw = distance.JaroWinkler.similarity(v1, v2) * 100.0
                n_lev = fuzz.ratio(v1, v2)
                n_sp = fuzz.ratio(s_v1, s_v2) if (s_v1 and s_v2) else 0.0
                l_rat = min(len(s_v1), len(s_v2)) / max(1, max(len(s_v1), len(s_v2)))

                strict_sim = max(n_sort, n_sp, 0.92 * n_jw)
                set_weight = 0.95 if l_rat >= 0.6 else 0.82
                var_score = max(strict_sim, set_weight * n_set)

                if n_sort > best_sort:
                    best_sort = n_sort
                if n_set > best_set:
                    best_set = n_set
                if n_jw > best_jw:
                    best_jw = n_jw
                if n_lev > best_lev:
                    best_lev = n_lev
                if n_sp > best_spaceless:
                    best_spaceless = n_sp
                if l_rat > best_len_ratio:
                    best_len_ratio = l_rat
                if var_score > best_name_score:
                    best_name_score = var_score
                if strict_sim > best_strict_name:
                    best_strict_name = strict_sim

        cand_coined = is_single_coined_alias(cand_rec.business_name, vars2[0])
        coined_flag = 1.0 if (s1_coined or cand_coined) else 0.0

        full_sort = fuzz.token_sort_ratio(full_n1, full_n2)

        t2_split = strip_n2.split()
        cand_sig_tokens = [
            t for t in t2_split
            if len(t) >= 2 and t not in GENERIC_CORP_WORDS
        ]
        cand_token_set = set(cand_sig_tokens)
        shared_name_tokens = s1_token_set & cand_token_set
        union_name_tokens = s1_token_set | cand_token_set
        name_jaccard = len(shared_name_tokens) / max(1, len(union_name_tokens))

        # Sister-company unmatched token conflict features (strict & full corporate discriminator sets)
        unm_s1 = _count_unmatched_tokens(s1_token_set, cand_token_set)
        unm_c = _count_unmatched_tokens(cand_token_set, s1_token_set)
        min_unm_toks = float(min(unm_s1, unm_c))
        max_unm_toks = float(max(unm_s1, unm_c))

        cand_full_list = [
            t for t in t2_split
            if len(t) >= 2 and t not in FULL_NAME_STOP_WORDS
        ]
        cand_full_tokens = set(cand_full_list)
        full_name_jacc = len(s1_full_tokens & cand_full_tokens) / max(1, len(s1_full_tokens | cand_full_tokens))
        full_unm_s1 = _count_unmatched_tokens(s1_full_tokens, cand_full_tokens)
        full_unm_c = _count_unmatched_tokens(cand_full_tokens, s1_full_tokens)
        full_min_unm = float(min(full_unm_s1, full_unm_c))
        full_max_unm = float(max(full_unm_s1, full_unm_c))

        # Check exact initials acronym alias & synthetic generic word substitution
        is_acronym_alias = False
        if len(t2_split) == 1 and 2 <= len(t2_split[0]) <= 4 and t2_split[0] in s1_inits:
            is_acronym_alias = True
        elif len(s1_nonstop_list) == 1 and 2 <= len(s1_nonstop_list[0]) <= 4 and len(cand_sig_tokens) >= 2:
            c_inits = {
                "".join(t[0] for t in cand_sig_tokens),
                "".join(t[0] for t in cand_full_list),
            }
            if s1_nonstop_list[0] in c_inits:
                is_acronym_alias = True

        # Credential or short-code conflict (e.g. DPM vs DUM/RPM/DPMB, MD vs GMD, VED vs JVED, QC vs KQC)
        s1_only_full = s1_full_tokens - cand_full_tokens
        cand_only_full = cand_full_tokens - s1_full_tokens
        is_syn_generic_sub = (
            bool(s1_full_tokens & cand_full_tokens)
            and (
                (bool(cand_only_full) and cand_only_full <= _SYN_GENERIC_SUB_WORDS and len(s1_only_full) == 1)
                or (bool(s1_only_full) and s1_only_full <= _SYN_GENERIC_SUB_WORDS and len(cand_only_full) == 1)
            )
        )

        cred_conflict = 0.0
        if (s1_only_full | cand_only_full) & _CORRUPTED_MED_TOKENS:
            cred_conflict = 1.0
        elif s1_only_full and cand_only_full:
            for t1 in s1_only_full:
                for t2 in cand_only_full:
                    if len(t1) <= 4 and len(t2) <= 4:
                        if (t1 in _MED_CRED_TOKENS or t2 in _MED_CRED_TOKENS):
                            cred_conflict = 1.0
                            break
                        if (t1 in t2 or t2 in t1) or fuzz.ratio(t1, t2) >= 60.0:
                            cred_conflict = 1.0
                            break
                if cred_conflict == 1.0:
                    break

        cand_first_word = cand_full_list[0] if cand_full_list else ""
        if s1_first_word and cand_first_word:
            if s1_first_word == cand_first_word:
                first_w_sim = 1.0
            elif s1_first_word[0] == cand_first_word[0] and min(len(s1_first_word), len(cand_first_word)) >= 4:
                first_w_sim = fuzz.ratio(s1_first_word, cand_first_word) / 100.0
            else:
                first_w_sim = 0.0
        else:
            first_w_sim = 0.0

        # Roman numeral / branch sequel conflict feature
        cand_branch_tokens = cand_full_raw_toks & BRANCH_OR_ROMAN_TOKENS
        branch_conflict = 1.0 if len(s1_branch_tokens ^ cand_branch_tokens) > 0 else 0.0

        if s1_sig_tokens and cand_sig_tokens:
            w1, w2 = s1_sig_tokens[0], cand_sig_tokens[0]
            prefix_match = 1.0 if (w1 == w2 or (len(w1) >= 4 and len(w2) >= 4 and w1[:4] == w2[:4])) else 0.0
        else:
            prefix_match = 0.0

        shared_idf_sum = 0.0
        if country_token_df is not None:
            for t in shared_name_tokens:
                df = country_token_df.get(t, 0)
                shared_idf_sum += 1.0 / math.log(2.0 + df)
        else:
            shared_idf_sum = float(len(shared_name_tokens)) * 0.25

        # Address features + Inlined Heuristic Score
        addr2 = clean_address(cand_rec.business_address, cand_rec.country)
        if addr2 and (s1_has_indic_addr != bool(INDIC_RANGE_RE.search(cand_rec.business_address))):
            addr2 = _align_cross_script_text(addr2, s1_addr_skel_map)
        cand_clean_addrs[idx] = addr2

        if not addr2:
            addr_missing = 1.0
            addr_sort = -1.0
            addr_set = -1.0
            addr_jw = -1.0
            street_rare_jacc = -1.0
            shared_md_nums = 0.0
            md_num_jacc = -1.0
            prim_num_match = 0.0
            md_conflict = 0.0
            prim_num_sim = -1.0
            all_num_jacc = -1.0
            both_unm_nums = 0.0
            unm_nums_cnt = 0.0
            unm_num_max_sim = -1.0
            loc_conflict = 0.0
            pure_st_sort = -1.0
            pure_st_set = -1.0
            min_pure_st_words = -1.0
            shared_3p_nums = 0.0
            unm_2p_nums_cnt = 0.0

            base_missing = 0.90 * best_strict_name
            if legal_conflict == 1.0 or cred_conflict == 1.0:
                base_missing -= 15.0
            if country_token_df is not None:
                if v1_0_sig_tokens:
                    if v1_0_min_df > 120 and len(v1_0_sig_tokens) <= 2:
                        base_missing -= 8.0
                    elif v1_0_min_df <= 25 and best_strict_name >= 95.0:
                        base_missing += 3.0
                else:
                    base_missing -= 12.0
            h_score = min(100.0, max(0.0, base_missing))
        else:
            addr_missing = 0.0
            addr_sort = fuzz.token_sort_ratio(addr1, addr2)
            addr_set = fuzz.token_set_ratio(addr1, addr2)
            addr_jw = distance.JaroWinkler.similarity(addr1, addr2) * 100.0

            cand_addr_split = addr2.split()
            cand_addr_tokens = set(cand_addr_split)
            cand_rare_addr_words = {
                t for t in cand_addr_tokens
                if len(t) >= 3 and not t.isdigit() and t not in ADDR_STOP_TOKENS
            }
            u_addr = s1_rare_addr_words | cand_rare_addr_words
            street_rare_jacc = len(s1_rare_addr_words & cand_rare_addr_words) / max(1, len(u_addr))

            cand_pure_street_list = [
                t for t in cand_addr_split
                if len(t) >= 3 and t.isalpha() and t not in NON_STREET_ADDR_TOKENS
            ]
            cand_pure_street_str = " ".join(cand_pure_street_list)
            min_pure_st_words = float(min(s1_pure_street_cnt, len(cand_pure_street_list)))
            if s1_pure_street_str and cand_pure_street_str:
                pure_st_sort = fuzz.token_sort_ratio(s1_pure_street_str, cand_pure_street_str)
                pure_st_set = fuzz.token_set_ratio(s1_pure_street_str, cand_pure_street_str)
            elif not s1_pure_street_str and not cand_pure_street_str:
                pure_st_sort = 50.0
                pure_st_set = 50.0
            else:
                pure_st_sort = 0.0
                pure_st_set = 0.0

            cand_loc_tokens = cand_addr_tokens & LOC_TOKENS
            if "telangana" in cand_loc_tokens or "andhra" in cand_loc_tokens:
                cand_loc_tokens = cand_loc_tokens | {"telangana", "andhra"}
            loc_conflict = 1.0 if (s1_loc_tokens and cand_loc_tokens and not (s1_loc_tokens & cand_loc_tokens)) else 0.0

            cand_nums = extract_multi_digit_numbers(addr2)
            cand_num_set = set(cand_nums)
            shared_nums_set = s1_num_set & cand_num_set
            shared_md_nums = float(len(shared_nums_set))
            cand_3plus_nums = {n for n in cand_num_set if len(n) >= 3}
            shared_3p_nums = float(len(s1_3plus_nums & cand_3plus_nums))

            if s1_num_set and cand_num_set:
                md_num_jacc = len(shared_nums_set) / len(s1_num_set | cand_num_set)
                if s1_nums[0] == cand_nums[0]:
                    prim_num_match = 1.0
                    prim_num_sim = 1.0
                elif (
                    ((len(s1_nums[0]) <= 4 or len(s1_nums) >= 2) and s1_nums[0] in cand_num_set)
                    or ((len(cand_nums[0]) <= 4 or len(cand_nums) >= 2) and cand_nums[0] in s1_num_set)
                ):
                    prim_num_match = 0.75
                    prim_num_sim = max(0.95, fuzz.ratio(s1_nums[0], cand_nums[0]) / 100.0)
                else:
                    prim_num_match = -1.0
                    prim_num_sim = fuzz.ratio(s1_nums[0], cand_nums[0]) / 100.0
                md_conflict = 1.0 if len(shared_nums_set) == 0 else 0.0
            else:
                md_num_jacc = 0.0
                prim_num_match = 0.0
                prim_num_sim = -1.0
                md_conflict = 0.0

            # Full-Pool All-Number (including 1-digit door/floor/sector & ordinals) Multiset Disambiguation
            cand_all_num_list = [m.lstrip("0") or "0" for m in ALL_DIGIT_RE.findall(addr2)]
            if s1_all_num_list and cand_all_num_list:
                c_cand = Counter(cand_all_num_list)
                cand_hyph_joins = {
                    (m1.lstrip("0") or "0") + m2
                    for m1, m2 in HYPH_NUM_RE.findall(cand_rec.business_address)
                } | {
                    (m1.lstrip("0") or "0") + (m2.lstrip("0") or "0")
                    for m1, m2 in HYPH_NUM_RE.findall(cand_rec.business_address)
                }
                inter_cnt = sum((s1_all_num_counter & c_cand).values())
                union_cnt = sum((s1_all_num_counter | c_cand).values())
                all_num_jacc = inter_cnt / max(1, union_cnt)

                u1 = [k for k in (s1_all_num_counter - c_cand) if k not in cand_hyph_joins]
                u2 = [k for k in (c_cand - s1_all_num_counter) if k not in s1_hyph_joins]
                if (set(s1_all_num_list) & cand_hyph_joins) or (set(cand_all_num_list) & s1_hyph_joins):
                    all_num_jacc = max(all_num_jacc, 0.85)
                    both_unm_nums = 0.0
                    unm_nums_cnt = 0.0
                    unm_2p_nums_cnt = 0.0
                    unm_num_max_sim = 1.0
                else:
                    unm_nums_cnt = float(len(u1) + len(u2))
                    unm_2p_nums_cnt = float(
                        sum(1 for k in u1 if len(k) >= 2) + sum(1 for k in u2 if len(k) >= 2)
                    )
                    if u1 and u2:
                        both_unm_nums = 1.0
                        is_trunc = any(
                            (
                                ((x in y or y in x) and min(len(x), len(y)) >= 2)
                                or ((x.endswith(y) or y.endswith(x)) and pure_st_set >= 80.0)
                            )
                            and len(x) != len(y)
                            for x in u1 for y in u2
                        )
                        is_pure_pm2 = (
                            len(u1) == 1
                            and len(u2) == 1
                            and abs(int(u1[0]) - int(u2[0])) <= 2
                            and pure_st_set >= 80.0
                            and addr_set >= 80.0
                            and best_sort >= 99.0
                            and full_max_unm == 0.0
                            and legal_conflict == 0.0
                        )
                        unm_num_max_sim = 0.85 if is_trunc else (0.75 if is_pure_pm2 else 0.0)
                    elif not u1 and not u2:
                        both_unm_nums = 0.0
                        unm_num_max_sim = 1.0
                    else:
                        both_unm_nums = 0.0
                        unm_num_max_sim = 0.5
            else:
                all_num_jacc = 0.0
                both_unm_nums = 0.0
                unm_nums_cnt = float(len(s1_all_num_list) + len(cand_all_num_list))
                unm_2p_nums_cnt = float(len(s1_nums) + len(cand_nums))
                unm_num_max_sim = -1.0

            cand_all_nums = extract_numbers(addr2)
            shared_all_nums = s1_all_nums & cand_all_nums
            if s1_all_nums and cand_all_nums:
                overlap = len(shared_all_nums) / max(1, min(len(s1_all_nums), len(cand_all_nums)))
                num_bonus = 6.0 * overlap - 5.0 * (1.0 - overlap)
            else:
                num_bonus = 0.0

            has_hard_num_conflict = (both_unm_nums == 1.0 and unm_num_max_sim == 0.0)
            if (
                has_hard_num_conflict
                and unm_2p_nums_cnt == 0.0
                and pure_st_set >= 85.0
                and any(2 <= len(n) <= 4 for n in shared_nums_set)
            ):
                has_hard_num_conflict = False
                unm_num_max_sim = 0.70
            if has_hard_num_conflict:
                num_bonus -= 14.0

            if not has_hard_num_conflict and loc_conflict == 0.0 and legal_conflict < 1.0:
                if is_acronym_alias and addr_set >= 80.0 and (len(shared_all_nums) >= 1 or not s1_all_nums):
                    coined_flag = 1.0
                    min_unm_toks = 0.0
                    full_min_unm = 0.0
                    cred_conflict = 0.0
                    best_sort = max(best_sort, 92.0)
                    best_set = max(best_set, 95.0)
                    best_name_score = max(best_name_score, 92.0)
                    best_strict_name = max(best_strict_name, 92.0)
                elif is_syn_generic_sub and addr_set >= 85.0 and cred_conflict == 0.0 and (len(shared_all_nums) >= 1 or not s1_all_nums):
                    min_unm_toks = 0.0
                    full_min_unm = 0.0
                    best_set = max(best_set, 88.0)
                    best_name_score = max(best_name_score, 85.0)
                elif coined_flag == 1.0 and (addr_sort >= 88.0 or addr_set >= 90.0) and len(shared_all_nums) >= 1:
                    min_unm_toks = 0.0
                    full_min_unm = 0.0

            addr_score = min(100.0, max(0.0, max(addr_sort, 0.93 * addr_set) + num_bonus))
            composite = 0.60 * best_name_score + 0.40 * addr_score

            if legal_conflict == 1.0:
                composite -= 16.0
            if cred_conflict == 1.0:
                composite -= 14.0
            if min_pure_st_words >= 1.0 and pure_st_set < 35.0:
                composite -= 10.0
            if full_min_unm >= 1.0:
                composite -= 10.0
            elif full_max_unm >= 1.0 and full_name_jacc < 0.75:
                composite -= 6.0

            if not has_hard_num_conflict and legal_conflict < 1.0 and cred_conflict == 0.0 and full_min_unm == 0.0:
                if best_strict_name >= 95.0 and full_max_unm == 0.0 and (len(shared_all_nums) >= 1 or addr_set >= 72.0):
                    composite = max(composite, 88.0)
                if best_name_score >= 58.0 and addr_score >= 84.0 and (len(shared_all_nums) >= 1 or not s1_all_nums or not cand_all_nums):
                    composite = max(composite, 83.5)

            if best_name_score < 45.0:
                if (
                    not has_hard_num_conflict
                    and (addr_sort >= 88.0 or addr_set >= 92.0)
                    and len(shared_all_nums) >= 1
                    and (cand_coined or s1_coined)
                ):
                    composite = max(composite, 83.5)
                else:
                    composite = min(composite, 72.0)
            h_score = min(100.0, max(0.0, composite))
            if s1_rec.country != cand_rec.country:
                h_score = 0.0

        feats[idx, 0] = best_sort
        feats[idx, 1] = best_set
        feats[idx, 2] = best_jw
        feats[idx, 3] = best_lev
        feats[idx, 4] = best_spaceless
        feats[idx, 5] = full_sort
        feats[idx, 6] = name_jaccard
        feats[idx, 7] = prefix_match
        feats[idx, 8] = best_len_ratio
        feats[idx, 9] = s1_word_count
        feats[idx, 10] = s1_min_df
        feats[idx, 11] = shared_idf_sum
        feats[idx, 12] = addr_missing
        feats[idx, 13] = addr_sort
        feats[idx, 14] = addr_set
        feats[idx, 15] = addr_jw
        feats[idx, 16] = street_rare_jacc
        feats[idx, 17] = shared_md_nums
        feats[idx, 18] = md_num_jacc
        feats[idx, 19] = prim_num_match
        feats[idx, 20] = md_conflict
        feats[idx, 21] = coined_flag
        feats[idx, 22] = h_score
        feats[idx, 26] = min_unm_toks
        feats[idx, 27] = max_unm_toks
        feats[idx, 28] = branch_conflict
        feats[idx, 29] = prim_num_sim
        feats[idx, 30] = full_name_jacc
        feats[idx, 31] = full_min_unm
        feats[idx, 32] = full_max_unm
        feats[idx, 33] = all_num_jacc
        feats[idx, 34] = both_unm_nums
        feats[idx, 35] = unm_nums_cnt
        feats[idx, 36] = unm_num_max_sim
        feats[idx, 37] = loc_conflict
        feats[idx, 38] = legal_conflict
        feats[idx, 39] = first_w_sim
        feats[idx, 40] = pure_st_sort
        feats[idx, 41] = pure_st_set
        feats[idx, 42] = min_pure_st_words
        feats[idx, 43] = shared_3p_nums
        feats[idx, 44] = unm_2p_nums_cnt
        feats[idx, 49] = cred_conflict

    # Group Consensus Features (columns 45, 46, 47, 48) + Group-Relative Ranking Features (columns 23, 24, 25)
    same_name_match_addr_cnt = 0.0
    same_name_diff_addr_cnt = 0.0
    is_self_match = np.zeros(n_cands, dtype=np.float32)
    is_self_diff = np.zeros(n_cands, dtype=np.float32)
    best_addr_idx = -1
    best_addr_h = -1.0
    valid_addr_indices: List[int] = []

    for i in range(n_cands):
        if feats[i, 12] == 0.0:
            if feats[i, 1] >= 55.0 or feats[i, 21] == 1.0:
                valid_addr_indices.append(i)
            if feats[i, 1] >= 65.0 and feats[i, 22] > best_addr_h:
                best_addr_h = float(feats[i, 22])
                best_addr_idx = i
            is_same_name = (
                (feats[i, 0] >= 78.0 or (feats[i, 1] >= 88.0 and feats[i, 8] >= 0.65))
                and feats[i, 31] == 0.0
                and feats[i, 38] < 1.0
                and feats[i, 49] == 0.0
            )
            is_strict_same_name = (
                (feats[i, 0] >= 84.0 or (feats[i, 1] >= 92.0 and feats[i, 8] >= 0.70))
                and feats[i, 32] == 0.0
                and feats[i, 38] < 1.0
                and feats[i, 49] == 0.0
            )
            if is_same_name and (
                feats[i, 37] == 0.0
                and feats[i, 44] == 0.0
                and (feats[i, 14] >= 72.0 or (feats[i, 17] >= 1.0 and feats[i, 41] >= 65.0))
            ):
                same_name_match_addr_cnt += 1.0
                is_self_match[i] = 1.0
            elif is_strict_same_name and (
                feats[i, 37] == 1.0
                or (feats[i, 14] < 58.0 and feats[i, 41] < 50.0 and feats[i, 17] == 0.0)
            ):
                same_name_diff_addr_cnt += 1.0
                is_self_diff[i] = 1.0

    feats[:, 45] = same_name_match_addr_cnt - is_self_match
    feats[:, 46] = same_name_diff_addr_cnt - is_self_diff

    # Boost heuristic score for sibling-confirmed missing-address candidates before computing group ranks
    for i in range(n_cands):
        if (
            feats[i, 12] == 1.0
            and feats[i, 0] >= 85.0
            and feats[i, 32] == 0.0
            and feats[i, 38] < 1.0
            and feats[i, 49] == 0.0
            and feats[i, 45] >= 1.0
            and feats[i, 46] == 0.0
        ):
            feats[i, 22] = max(feats[i, 22], 91.0)

    h_scores = feats[:, 22]
    max_h = float(np.max(h_scores))
    order = np.argsort(-h_scores)
    ranks = np.empty(n_cands, dtype=np.float32)
    ranks[order] = np.arange(1, n_cands + 1, dtype=np.float32)

    addr_sorts = feats[:, 13]
    valid_addr_sorts = addr_sorts[addr_sorts >= 0.0]
    max_addr_sort = float(np.max(valid_addr_sorts)) if len(valid_addr_sorts) > 0 else 0.0

    feats[:, 23] = ranks
    feats[:, 24] = max_h - h_scores
    feats[:, 25] = np.where(addr_sorts >= 0.0, max_addr_sort - addr_sorts, -1.0)

    top_addr_str = cand_clean_addrs[best_addr_idx] if best_addr_idx >= 0 else ""
    for i in range(n_cands):
        if feats[i, 12] == 1.0:
            feats[i, 47] = 0.0
            feats[i, 48] = -1.0
        else:
            a_i = cand_clean_addrs[i]
            feats[i, 48] = fuzz.token_sort_ratio(a_i, top_addr_str) if top_addr_str else -1.0
            c_cnt = 0.0
            for j in valid_addr_indices:
                if j != i and fuzz.token_sort_ratio(a_i, cand_clean_addrs[j]) >= 82.0:
                    c_cnt += 1.0
            feats[i, 47] = c_cnt

    return feats
