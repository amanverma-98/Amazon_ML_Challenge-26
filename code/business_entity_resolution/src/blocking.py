"""
Blocking / Candidate Generation module for Business Entity Resolution (EXP-002).
Implements Country-Partitioned Multi-Granularity Tokenizer & Inverted Index:
- Spaceless concatenated name keys ('ns:')
- DBA-aware word token keys ('nt:')
- Sorted 2-word shingle keys ('sh:') to rescue common industry words
- 5-char subword n-gram keys ('ng:') for compound words and typos
- Address number+word ('an:'), address word-pair ('aw:'), and postal code ('pc:') keys
"""

import heapq
import math
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set

from code.business_entity_resolution.src.data_loader import EntityRecord
from code.business_entity_resolution.src.preprocessor import (
    _SKEL_REPEAT_RE,
    clean_address,
    extract_name_variants,
    phonetic_skeleton_word,
)

PURE_STOP_WORDS = {
    "the", "and", "of", "in", "at", "for", "to", "on", "by", "with", "de", "du", "des", "la", "le", "les",
    "inc", "llc", "ltd", "pvt", "private", "limited", "corp", "corporation",
    "co", "company", "llp", "pllc", "plc", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "gmbh",
    "dba", "aka", "smt", "shri", "sri", "ms", "mr", "mrs", "dr", "messrs",
}

GENERIC_CORP_WORDS = PURE_STOP_WORDS | {
    "group", "services", "service", "solutions", "enterprises", "enterprise", "holdings",
    "international", "global", "india", "usa", "us", "france", "center", "centre",
    "partners", "partner",
}

STATE_TOKENS = {
    "uttar", "pradesh", "maharashtra", "delhi", "karnataka", "tamil", "nadu", "gujarat",
    "madhya", "haryana", "rajasthan", "telangana", "andhra", "kerala", "punjab", "bengal",
    "bihar", "odisha", "jharkhand", "chhattisgarh", "uttarakhand", "himachal", "assam", "goa",
    "california", "texas", "florida", "illinois", "pennsylvania", "ohio", "georgia",
    "michigan", "carolina", "jersey", "virginia", "washington", "arizona", "tennessee",
    "indiana", "missouri", "maryland", "wisconsin", "colorado", "minnesota", "alabama",
    "kentucky", "oregon", "oklahoma", "connecticut", "utah", "massachusetts", "nevada",
    "hawaii", "idaho", "iowa", "kansas", "louisiana", "maine", "mississippi", "montana",
    "nebraska", "hampshire", "mexico", "dakota", "arkansas", "delaware", "rhode", "vermont",
    "wyoming", "alaska", "columbia",
    "nouvelle", "aquitaine", "auvergne", "rhone", "alpes", "bourgogne", "franche", "comte",
    "bretagne", "loire", "corse", "occitanie", "normandie", "provence", "azur",
    "gironde", "calais", "atlantique", "maritime", "maritimes", "bouches", "rhin", "garonne",
    "isere", "moselle", "finistere", "oise", "morbihan", "herault", "var", "seine", "marne",
    "yvelines", "essonne",
}

ADDR_STOP_TOKENS = {
    "road", "street", "avenue", "boulevard", "lane", "drive", "court", "place", "allee", "rue",
    "ter", "quater", "impasse", "mail", "faubourg", "cours", "chemin", "route", "quai", "square",
    "passage", "voie", "esplanade", "promenade", "cedex", "parkway", "highway",
    "floor", "ground", "first", "second", "third", "fourth", "fifth", "unit", "suite", "apartment",
    "shop", "no", "plot", "flat", "sector", "phase", "block", "near", "opposite",
    "behind", "beside", "main", "cross", "stage", "layout", "colony", "nagar",
    "vihar", "marg", "puram", "khand", "peth", "bazar", "market", "plaza",
    "building", "bldg", "complex", "tower", "towers", "center", "centre", "park", "circle",
    "north", "south", "east", "west", "new", "old", "city", "state", "district",
    "suburban", "urban", "rural", "tehsil", "taluka", "taluk", "mandal", "township", "county", "borough", "ward",
    "room", "hno", "dno", "survey", "khasra", "extension", "extn", "ext",
    "po", "box", "post", "office", "pin", "zip", "de", "du", "des", "la", "le", "les", "bis",
    "pradesh", "nadu", "hauts", "pays", "grand", "ile", "val", "cote",
}

CITY_TOKENS = {
    "mumbai", "bangalore", "bengaluru", "kolkata", "chennai", "madras", "hyderabad",
    "pune", "ahmedabad", "surat", "jaipur", "lucknow", "kanpur", "nagpur", "indore",
    "thane", "bhopal", "visakhapatnam", "patna", "vadodara", "ghaziabad", "ludhiana",
    "agra", "nashik", "faridabad", "meerut", "rajkot", "varanasi", "srinagar",
    "aurangabad", "dhanbad", "amritsar", "allahabad", "ranchi", "howrah", "coimbatore",
    "jabalpur", "gwalior", "vijayawada", "jodhpur", "madurai", "raipur", "kota",
    "guwahati", "chandigarh", "solapur", "hubli", "mysore", "tiruchirappalli",
    "bareilly", "aligarh", "tiruppur", "gurgaon", "gurugram", "noida", "kochi",
    "ernakulam", "trivandrum", "thiruvananthapuram", "bhubaneswar", "dehradun",
    "paris", "lyon", "marseille", "toulouse", "nice", "nantes", "strasbourg",
    "montpellier", "bordeaux", "lille", "rennes", "reims", "toulon", "grenoble",
    "dijon", "angers", "nimes", "villeurbanne", "clermont", "ferrand", "aix", "brest",
    "tours", "amiens", "limoges", "annecy", "perpignan", "metz", "besancon", "orleans",
    "rouen", "mulhouse", "caen", "nancy", "nanterre", "avignon", "creteil", "poitiers",
    "versailles", "pau", "rochelle", "cannes", "antibes", "dunkerque", "beziers",
    "colmar", "bourges", "merignac", "ajaccio", "quimper", "valence", "troyes",
    "chambery", "lorient", "niort", "montauban",
    "chicago", "houston", "phoenix", "philadelphia", "dallas", "austin", "atlanta",
    "miami", "seattle", "denver", "boston", "detroit", "portland", "memphis",
    "baltimore", "milwaukee", "albuquerque", "tucson", "fresno", "sacramento",
}

_DIGIT_RUN_RE = re.compile(r"\d+")
_ALPHA_RUN_RE = re.compile(r"[a-z]{3,}")


def extract_blocking_keys(record: EntityRecord) -> List[str]:
    """
    Extracts multi-granularity blocking keys for a record.
    """
    keys: Set[str] = set()
    variants = extract_name_variants(record.business_name)
    primary_skel_full = ""
    primary_first_key = ""

    for v_idx, stripped_name in enumerate(variants):
        raw_tokens = [t for t in stripped_name.split() if t not in PURE_STOP_WORDS]
        sig_tokens = [t for t in raw_tokens if len(t) >= 2 and t not in GENERIC_CORP_WORDS]

        # 1. Spaceless concatenated name (catches URLs & joined words like 'pushorizonpaper')
        spaceless = "".join(sig_tokens)
        if len(spaceless) >= 4:
            keys.add(f"ns:{spaceless}")
            # 2. Subword 5-char n-grams on spaceless string (catches compound words & typos)
            if len(spaceless) >= 5:
                step = 1 if len(spaceless) <= 10 else 2
                for i in range(0, min(len(spaceless) - 4, 12), step):
                    keys.add(f"ng:{spaceless[i:i+5]}")

        # 3. Phonetic Consonant Skeleton (bridges English vs Indic transliterations & typos)
        skel_tokens = []
        for tok in sig_tokens[:5]:
            keys.add(f"nt:{tok}")
            if len(tok) >= 4 and not tok.isdigit():
                keys.add(f"np:{tok[:4]}")
            if not tok.isdigit():
                sk = phonetic_skeleton_word(tok)
                if sk:
                    skel_tokens.append(sk)
                    if len(sk) >= 4 or (len(sig_tokens) == 1 and len(sk) >= 3):
                        keys.add(f"ph:{sk}")

        if skel_tokens:
            skel_full = _SKEL_REPEAT_RE.sub(r"\1", "".join(skel_tokens))
            if len(skel_full) >= 3:
                keys.add(f"ps:{skel_full}")
            if v_idx == 0:
                primary_skel_full = skel_full
                primary_first_key = skel_tokens[0] if len(skel_tokens[0]) >= 2 else (sig_tokens[0] if sig_tokens else "")

        # 4. Sorted 2-word Shingles (uses raw_tokens so pairs like 'cardiology_center' or 'clinic_highland' are indexed!)
        shingle_tokens = sorted(set(t for t in raw_tokens if len(t) >= 3))[:5]
        if len(shingle_tokens) >= 2:
            for w1, w2 in combinations(shingle_tokens, 2):
                keys.add(f"sh:{w1}_{w2}")

    # 5. Address Keys & Compound Name+Location Keys (nc:)
    if record.business_address:
        cleaned_addr = clean_address(record.business_address, record.country)
        addr_tokens = cleaned_addr.split()
        nums: List[str] = []
        non_city_words: List[str] = []
        city_words: List[str] = []
        state_words: List[str] = []

        for tok in addr_tokens:
            if tok.isdigit():
                if len(tok) >= 2 or not nums:
                    if tok not in nums:
                        nums.append(tok)
            elif tok.isalpha():
                if tok in STATE_TOKENS:
                    if tok not in state_words:
                        state_words.append(tok)
                elif len(tok) >= 3 and tok not in ADDR_STOP_TOKENS:
                    if tok in CITY_TOKENS:
                        if tok not in city_words:
                            city_words.append(tok)
                    elif tok not in non_city_words:
                        non_city_words.append(tok)
            else:
                # Alphanumeric token (e.g., '16a', '10syno246', 'd57', '2e')
                if len(tok) <= 5 and tok not in nums:
                    nums.append(tok)
                for d in _DIGIT_RUN_RE.findall(tok):
                    d_norm = d.lstrip("0") or "0"
                    if (len(d_norm) >= 2 or not nums) and d_norm not in nums:
                        nums.append(d_norm)
                for a in _ALPHA_RUN_RE.findall(tok):
                    if a not in ADDR_STOP_TOKENS and a not in STATE_TOKENS:
                        if a in CITY_TOKENS:
                            if a not in city_words:
                                city_words.append(a)
                        elif a not in non_city_words:
                            non_city_words.append(a)

        # Anchor words include BOTH head (building/street) and tail (district/city/state) tokens!
        anchor_words: List[str] = []
        for w in non_city_words[:3] + non_city_words[-2:] + city_words[:2] + state_words[:1]:
            if w not in anchor_words:
                anchor_words.append(w)

        for num in nums[:3]:
            for w in anchor_words[:6]:
                keys.add(f"an:{num}_{w}")
            if len(num) >= 5 and num.isdigit():
                keys.add(f"pc:{num}")

        # Compound Name + City/State Keys (breaks ties among 200+ same-named companies nationwide!)
        locs: List[str] = []
        for w in city_words[:2] + state_words[:1] + non_city_words[-1:]:
            if w not in locs:
                locs.append(w)
        for loc in locs[:3]:
            if len(primary_skel_full) >= 3:
                keys.add(f"nc:{primary_skel_full}_{loc}")
            if len(primary_first_key) >= 2 and primary_first_key != primary_skel_full:
                keys.add(f"nc:{primary_first_key}_{loc}")

        # High-precision Multi-Token Address Fingerprints (survives 50-Lakh pools for coined aliases!)
        distinct_non_city = sorted(non_city_words[:3])
        if len(distinct_non_city) >= 2 and nums:
            n0 = nums[0]
            for w1, w2 in combinations(distinct_non_city, 2):
                keys.add(f"af:{n0}_{w1}_{w2}")

        # Address rare word pairs (when street number is missing in one source)
        aw_pool: List[str] = []
        for w in non_city_words[:3] + non_city_words[-1:] + city_words[:1]:
            if w not in aw_pool:
                aw_pool.append(w)
        distinct_addr_words = sorted(aw_pool[:5])
        if len(distinct_addr_words) >= 2:
            for w1, w2 in combinations(distinct_addr_words, 2):
                keys.add(f"aw:{w1}_{w2}")

    return list(keys)


class CountryPartitionedBlocker:
    """
    Builds a country-partitioned inverted index over Source 2 and Source 3 records,
    and retrieves top-K candidate IDs per Source 1 entity using Multi-Channel Synergy & Quota scoring.
    Also exposes token document frequencies for downstream matcher precision guards.
    """

    def __init__(
        self,
        max_postings_per_key: int = 2500,
        top_k: int = 35,
    ):
        self.max_postings_per_key = max_postings_per_key
        self.top_k = top_k
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.key_weights: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.token_df: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def fit(
        self,
        s2_records: Dict[str, EntityRecord],
        s3_records: Dict[str, EntityRecord],
    ) -> None:
        """
        Indexes all S2 and S3 records partitioned by country with:
          1. Key-type-aware posting limits and immediate in-loop memory freeing.
          2. Scale-invariant normalization of token_df to the 234k reference pool size,
             and steep quadratic log-IDF weighting for candidate ranking.
        """
        max_p = self.max_postings_per_key
        high_prec_limit = max(max_p, 8000)   # ns:, ps:, nc:, sh:, af:, an: (high-precision compound keys)
        mid_prec_limit = min(max_p, 3500)    # nt:, ph:, aw:, pc: (word tokens & postal codes)
        sub_limit = min(max_p, 750)          # ng:, np: (broad subword 5-grams & 4-char prefixes)

        pruned: Dict[str, Set[str]] = defaultdict(set)
        country_doc_count: Dict[str, int] = defaultdict(int)

        for records_dict in (s2_records, s3_records):
            if not records_dict:
                continue
            for eid, rec in records_dict.items():
                country = rec.country
                country_doc_count[country] += 1
                c_idx = self.index[country]
                c_tdf = self.token_df[country]
                c_pruned = pruned[country]
                for key in extract_blocking_keys(rec):
                    if key.startswith("nt:"):
                        c_tdf[key[3:]] += 1
                    if key in c_pruned:
                        continue
                    lst = c_idx[key]
                    if key.startswith(("ng:", "np:")):
                        lim = sub_limit
                    elif key.startswith(("ns:", "ps:", "nc:", "sh:", "af:", "an:")):
                        lim = high_prec_limit
                    else:
                        lim = mid_prec_limit

                    if len(lst) >= lim:
                        lst.clear()
                        del c_idx[key]
                        c_pruned.add(key)
                    else:
                        lst.append(eid)

        pruned.clear()

        # Normalize token_df to the 234,392 reference pool scale so features
        # (s1_name_min_df, shared_name_idf_sum, heuristic_score) are 100% invariant to pool size!
        ref_pool_size = 234_392.0
        for country, c_idx in self.index.items():
            doc_cnt = max(1, country_doc_count[country])
            scale = min(1.0, ref_pool_size / float(doc_cnt))

            if scale < 0.95:
                c_tdf = self.token_df[country]
                for tok, raw_df in list(c_tdf.items()):
                    c_tdf[tok] = max(1, int(round(raw_df * scale)))

            c_weights = self.key_weights[country]
            for key, postings in c_idx.items():
                df = len(postings)
                prefix = key[:3]
                if prefix in ("ns:", "nc:"):
                    base_w = 3.5  # Exact spaceless name match & Name+Location compound key
                elif prefix in ("ps:", "af:"):
                    base_w = 2.8  # Exact phonetic skeleton & 3-token address fingerprint
                elif prefix in ("sh:", "an:"):
                    base_w = 2.2  # Word-pair shingle & address number+word anchor
                elif prefix in ("nt:", "aw:", "pc:"):
                    base_w = 1.5  # Single name word, address word-pair, postal code
                elif prefix == "ph:":
                    base_w = 1.25 # Phonetic word skeleton
                elif prefix == "ng:":
                    base_w = 0.35 # Subword 5-gram
                else:
                    base_w = 0.45 # 4-char prefix
                c_weights[key] = base_w / (math.log(2.0 + df) ** 2.0)

    def query_entity(self, s1_record: EntityRecord) -> List[str]:
        """
        Retrieves top-K candidate entity IDs from S2/S3 for a single S1 record using
        Multi-Channel Cross-Field Synergy & Guaranteed Quotas (Synergy, Name, Skeleton, Address).
        """
        country = s1_record.country
        c_idx = self.index.get(country)
        if not c_idx:
            return []

        c_weights = self.key_weights[country]
        name_scores: Dict[str, float] = defaultdict(float)
        skel_scores: Dict[str, float] = defaultdict(float)
        addr_scores: Dict[str, float] = defaultdict(float)

        for key in extract_blocking_keys(s1_record):
            postings = c_idx.get(key)
            if not postings:
                continue
            w = c_weights[key]
            prefix = key[:3]
            if prefix in ("af:", "an:", "aw:", "pc:"):
                for cand_id in postings:
                    addr_scores[cand_id] += w
            elif prefix == "nc:":
                for cand_id in postings:
                    name_scores[cand_id] += w
                    skel_scores[cand_id] += w
                    addr_scores[cand_id] += w
            else:
                for cand_id in postings:
                    name_scores[cand_id] += w
                    if prefix in ("ps:", "ph:"):
                        skel_scores[cand_id] += w

        if not name_scores and not addr_scores:
            return []

        all_cands = set(name_scores.keys()) | set(addr_scores.keys())
        if len(all_cands) <= self.top_k:
            return [
                cid for cid, _ in sorted(
                    ((c, name_scores.get(c, 0.0) + addr_scores.get(c, 0.0)) for c in all_cands),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]

        synergy_scores: Dict[str, float] = {}
        for cid in all_cands:
            ns = name_scores.get(cid, 0.0)
            ads = addr_scores.get(cid, 0.0)
            bonus = 0.50 if (ns > 0.0 and ads > 0.0) else 0.0
            synergy_scores[cid] = ns + 0.6 * ads + bonus

        k = self.top_k
        top_syn = [cid for cid, _ in heapq.nlargest(k, synergy_scores.items(), key=lambda x: x[1])]
        top_name = [cid for cid, _ in heapq.nlargest(12, name_scores.items(), key=lambda x: x[1])] if name_scores else []
        top_skel = [cid for cid, _ in heapq.nlargest(10, skel_scores.items(), key=lambda x: x[1])] if skel_scores else []
        top_addr = [cid for cid, _ in heapq.nlargest(10, addr_scores.items(), key=lambda x: x[1])] if addr_scores else []

        selected: List[str] = []
        seen: Set[str] = set()

        def add_from(lst: List[str], limit: int) -> None:
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

    def generate_candidates(
        self,
        s1_records: Dict[str, EntityRecord],
    ) -> Dict[str, List[str]]:
        """Generates candidate lists for all S1 records."""
        results: Dict[str, List[str]] = {}
        for s1_id, rec in s1_records.items():
            results[s1_id] = self.query_entity(rec)
        return results
