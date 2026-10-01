"""
Preprocessor and text normalization module for Business Entity Resolution (EXP-002).
Includes:
- Hindi Devanagari -> Latin phonetic & lexical transliteration
- URL / Domain stripping (.com, .net, .org, www.)
- Dotted/slashed acronym collapsing (D/U -> du, S.A.S. -> sas)
- DBA ('d/b/a', 'a/k/a') multi-variant name extraction
- US, India, and France legal suffix stripping
- Country-aware state and street abbreviation standardization
- Ordinal and leading-zero number normalization
"""

import re
import unicodedata
from functools import lru_cache
from typing import List, Tuple

# ---------------------------------------------------------------------------
# 1. Universal Indic (All 9 Brahmic Scripts: \u0900-\u0D7F) -> Latin Transliteration
# ---------------------------------------------------------------------------
INDIC_RANGE_RE = re.compile(r"[\u0900-\u0D7F]")

# Malayalam chillu consonants, Gurmukhi Tippi/Addak, Odia/Assamese WA/RA & Tamil aaytham direct mappings
SPECIAL_INDIC_CHARS = {
    "\u0D7A": "n",
    "\u0D7B": "n",
    "\u0D7C": "r",
    "\u0D7D": "l",
    "\u0D7E": "l",
    "\u0D7F": "k",
    "\u0B83": "f",
    "\u0983": "",
    "\u0A70": "\u0902",  # Gurmukhi Tippi -> Devanagari Anusvara
    "\u0A71": "",        # Gurmukhi Addak (gemination)
    "\u0B71": "v",       # Odia WA
    "\u09F0": "r",       # Assamese RA
    "\u09F1": "v",       # Assamese WA
    "\u0970": "",
    "\u0971": "",
    "\u200c": "",
    "\u200d": "",
}


def normalize_indic_to_devanagari(word: str) -> str:
    """
    Maps any Indic Brahmic script character (Devanagari, Bengali, Gurmukhi, Gujarati,
    Odia, Tamil, Telugu, Kannada, Malayalam in U+0900..U+0D7F) to its aligned
    Devanagari codepoint (U+0900 + (cp & 0x7F)) using Unicode's ISCII block layout.
    Includes phonological adjustments for Malayalam alveolar 't' (റ്റ / ്റ) and Tamil 'f'/'s'.
    """
    if "\u0D31" in word or "\u0D02" in word:
        word = word.replace("\u0D31\u0D4D\u0D31", "\u091F")
        word = word.replace("\u0D4D\u0D31", "\u094D\u091F")
        if word.endswith("\u0D02"):
            word = word[:-1] + "m"
    if "\u0B83" in word or "\u0B9A" in word:
        word = word.replace("\u0B83\u0BAA", "\u092B")
        word = _TAMIL_SINGLE_CHA_RE.sub("\u0938", word)

    out = []
    for ch in word:
        if ch in SPECIAL_INDIC_CHARS:
            out.append(SPECIAL_INDIC_CHARS[ch])
            continue
        cp = ord(ch)
        if 0x0980 <= cp <= 0x0D7F:
            offset = cp & 0x7F
            # Short e/o in Southern scripts map to standard long e/o in Devanagari
            if offset in (0x0E, 0x12, 0x46, 0x4A):
                offset += 1
            elif offset == 0x29:  # Dravidian alveolar na -> dental na (0x28)
                offset = 0x28
            elif offset == 0x31:  # Dravidian hard ra/rra -> ra (0x30)
                offset = 0x30
            elif offset == 0x34:  # Dravidian zha -> la (0x32)
                offset = 0x32
            elif offset in (0x5C, 0x5D):  # Bengali/Indic rra/rha -> da/dha
                offset = 0x21
            elif offset == 0x5F:  # Bengali/Odia YYA -> Devanagari YA (0x2F)
                offset = 0x2F
            out.append(chr(0x0900 + offset))
        else:
            out.append(ch)
    return "".join(out)


_TAMIL_SINGLE_CHA_RE = re.compile(r"(?<!\u0BCD)\u0B9A(?!\u0BCD\u0B9A)")
_LABIAL_CONSONANTS = {"प", "फ", "ब", "भ", "म"}

HINDI_CORP_LEXICON = {
    "प्राइवेट": "private",
    "प्राईवेट": "private",
    "लिमिटेड": "limited",
    "एलएलपी": "llp",
    "मार्केटिंग": "marketing",
    "एंटरप्राइजेज": "enterprises",
    "इंटरप्राइजेज": "enterprises",
    "एंटरप्राइज": "enterprise",
    "सर्विसेज": "services",
    "सर्विस": "service",
    "सॉल्यूशंस": "solutions",
    "सोल्यूशंस": "solutions",
    "प्रॉपर्टीज": "properties",
    "प्रोपर्टीज": "properties",
    "इंडिया": "india",
    "ट्रेडिंग": "trading",
    "ट्रेडर्स": "traders",
    "कंपनी": "company",
    "कम्पनी": "company",
    "ग्रुप": "group",
    "इंफ्रास्ट्रक्चर": "infrastructure",
    "इंफ्रा": "infra",
    "टेक्नोलॉजीज": "technologies",
    "टेक्नोलॉजी": "technology",
    "कंसल्टेंट्स": "consultants",
    "कंसल्टेंसी": "consultancy",
    "एसोसिएट्स": "associates",
    "लॉजिस्टिक्स": "logistics",
    "एक्सपोर्ट्स": "exports",
    "एक्सपोर्ट": "export",
    "इंपोर्ट्स": "imports",
    "फाइनेंस": "finance",
    "मोटर्स": "motors",
    "फार्मा": "pharma",
    "फार्मास्युटिकल्स": "pharmaceuticals",
    "केमिकल्स": "chemicals",
    "टेक्सटाइल्स": "textiles",
    "इंडस्ट्रीज": "industries",
    "प्रोजेक्ट्स": "projects",
    "प्रोजेक्ट": "project",
    "बिल्डर्स": "builders",
    "डेवलपर्स": "developers",
    "होटल": "hotel",
    "हॉस्पिटल": "hospital",
    "हॉस्पिटैलिटी": "hospitality",
    "अल्फा": "alpha",
    "बीटा": "beta",
    "डेल्टा": "delta",
    "ग्लोबल": "global",
    "इंटरनेशनल": "international",
    "नेशनल": "national",
    "रियल": "real",
    "अरिहंत": "arihant",
    "कंस्ट्रक्शंस": "constructions",
    "कंस्ट्रक्शन": "construction",
    "आनंद": "anand",
    "इंफोटेक": "infotech",
    "इन्फोटेक": "infotech",
    "महाराष्ट्र": "maharashtra",
    "दिल्ली": "delhi",
    "मुंबई": "mumbai",
    "ओं": "om",
    "ॐ": "om",
    "ओम": "om",
    "न्यू": "new",
    "सदर्न": "southern",
    "नॉर्दर्न": "northern",
    "वेस्टर्न": "western",
    "ईस्टर्न": "eastern",
    "मॉडर्न": "modern",
    "फॉर्च्यून": "fortune",
    "फूड्स": "foods",
    "फूड": "food",
    "लोटस": "lotus",
    "सिस्टम्स": "systems",
    "सिस्टम": "system",
    "सॉफ्टवेयर": "software",
    "हार्डवेयर": "hardware",
    "इंजीनियरिंग": "engineering",
    "केयर": "care",
    "मेडिकल": "medical",
    "हेल्थकेयर": "healthcare",
    "आईटी": "it",
    "एसएस": "ss",
}

DEV_VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}

DEV_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    "स़": "sh", "य़": "y", "ऩ": "n", "ऱ": "r", "ल़": "l",
}

DEV_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ॉ": "o", "ॅ": "e",
}

VIRAMA = "्"
ANUSVARA = ("ं", "ँ")


def transliterate_devanagari_word(word: str) -> str:
    """Converts a single Devanagari token to Latin script."""
    if word in HINDI_CORP_LEXICON:
        return HINDI_CORP_LEXICON[word]

    out = []
    n = len(word)
    i = 0
    while i < n:
        ch = word[i]
        # Handle consonant + nukta safely
        if i + 1 < n and word[i + 1] == "़":
            comb = ch + "़"
            if comb in DEV_CONSONANTS:
                ch = comb
            i += 1

        if ch in DEV_VOWELS:
            out.append(DEV_VOWELS[ch])
        elif ch in DEV_CONSONANTS:
            base = DEV_CONSONANTS[ch]
            # Look ahead for matra or virama
            if i + 1 < n:
                nxt = word[i + 1]
                if nxt == VIRAMA:
                    out.append(base)
                    i += 2
                    continue
                elif nxt in DEV_MATRAS:
                    out.append(base + DEV_MATRAS[nxt])
                    i += 2
                    continue
            # Word-final consonant in Hindi drops inherent schwa 'a'
            if i == n - 1:
                out.append(base)
            else:
                out.append(base + "a")
        elif ch in DEV_MATRAS:
            out.append(DEV_MATRAS[ch])
        elif ch in ANUSVARA:
            if i + 1 < n and word[i + 1] in _LABIAL_CONSONANTS:
                out.append("m")
            else:
                out.append("n")
        elif ch != VIRAMA and ord(ch) < 128:
            out.append(ch)
        i += 1
    return "".join(out)


def transliterate_if_needed(text: str) -> str:
    """Transliterates any Indic Brahmic script words (U+0900..U+0D7F) into Latin equivalents."""
    if not INDIC_RANGE_RE.search(text):
        return text
    dev_text = normalize_indic_to_devanagari(text)
    tokens = dev_text.split()
    converted = [
        transliterate_devanagari_word(tok) if INDIC_RANGE_RE.search(tok) else tok
        for tok in tokens
    ]
    return " ".join(converted)


# ---------------------------------------------------------------------------
# 2. Domain / URL, Initials, DBA & Legal Suffix Patterns
# ---------------------------------------------------------------------------
DBA_SPLIT_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.|dba|a\s*/\s*k\s*/\s*a|aka)\b", re.IGNORECASE)
URL_PREFIX_RE = re.compile(r"\b(?:https?://)?(?:www\.)?", re.IGNORECASE)
DOMAIN_SUFFIX_RE = re.compile(r"\.(?:com|net|org|co\.in|in|fr|us|io|biz|info|edu|gov)\b", re.IGNORECASE)
CONCAT_COM_RE = re.compile(r"^([a-z0-9]{6,})com$")
CONCAT_LEGAL_SUFFIX_RE = re.compile(r"\b([a-z0-9]{2,}?)(?:privatelimited|publiclimited|pvtltd|private|limited|limatid|public)\b")
LONG_REG_NUM_RE = re.compile(r"\b\d{7,}\b")

# Matches dotted or slashed initials like "D/U", "B.S.", "S.A.S."
INITIALS_3_RE = re.compile(r"\b([a-zA-Z])\s*[./]\s*([a-zA-Z])\s*[./]\s*([a-zA-Z])\b")
INITIALS_2_RE = re.compile(r"\b([a-zA-Z])\s*[./]\s*([a-zA-Z])\b")

LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b("
    r"private\s+limited|pvt\s+ltd|pvt\s+limited|p\s+ltd|pra\s+li|"
    r"piraivet\s+limitet|praibhet\s+limited|praivet\s+limited|praivat\s+limitad|praivarr\s+limirrad|private\s+limatid|"
    r"private|pvt|praivet|piraivet|praibhet|praivat|praivarr|"
    r"corporation|corp|incorporated|inc|"
    r"limited\s+liability\s+partnership|llp|elelpi|ailaailapi|ailailpi|"
    r"professional\s+limited\s+liability\s+company|pllc|"
    r"limited\s+liability\s+company|llc|"
    r"limited\s+partnership|lp|"
    r"public\s+limited\s+company|plc|"
    r"limited|ltd|limitet|limitad|limirrad|limatid|"
    r"societe\s+a\s+responsabilite\s+limitee|sarl|"
    r"societe\s+par\s+actions\s+simplifiee\s+unipersonnelle|sasu|"
    r"societe\s+par\s+actions\s+simplifiee|sas|"
    r"entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|eurl|"
    r"societe\s+civile\s+immobiliere|sci|"
    r"societe\s+en\s+nom\s+collectif|snc|"
    r"societe\s+anonyme|sa|"
    r"et\s+fils|and\s+fils|fils|et\s+freres|and\s+freres|freres|cie|groupe|"
    r"company|co"
    r")\b",
    re.IGNORECASE,
)

HONORIFIC_PREFIX_RE = re.compile(r"^(?:shri|sri|smt|messrs|ms|mr|mrs|dr)\s+", re.IGNORECASE)
_ALNUM_SPLIT_RE = re.compile(r"\d+|[a-z]+")
_TWO_ALPHA_RE = re.compile(r"[a-z]{2,}")
_OCR_WORD_DIGIT_TRANS = str.maketrans({"0": "o", "1": "i", "5": "s", "6": "g", "8": "b"})
_ZW_CHARS_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_DUP_BIGRAM_RE = re.compile(r"\b([a-z0-9]+\s+[a-z0-9]+)(?:\s+\1\b)+")
_DUP_WORD_RE = re.compile(r"\b([a-z0-9]{2,})(?:\s+\1\b)+")

# ---------------------------------------------------------------------------
# 3. Address & State Standardization (US, India, France)
# ---------------------------------------------------------------------------
ORDINAL_NUM_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
LEADING_ZERO_NUM_RE = re.compile(r"\b0+(\d+)\b")

ADDR_ABBR_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pkwy": "parkway",
    "hwy": "highway",
    "cir": "circle",
    "plz": "plaza",
    "flr": "floor",
    "fl": "floor",
    "flfor": "floor",
    "ste": "suite",
    "apt": "apartment",
    "ctr": "center",
    "centre": "center",
    "opp": "opposite",
    "nr": "near",
    "sec": "sector",
    "dist": "district",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "bengaluru": "bangalore",
    "bangaluru": "bangalore",
    "gurugram": "gurgaon",
    "trivandrum": "thiruvananthapuram",
    "cochin": "kochi",
    "baroda": "vadodara",
    "poona": "pune",
    "mysuru": "mysore",
    "mangaluru": "mangalore",
    "orissa": "odisha",
    "keralam": "kerala",
    "grater": "greater",
    "steret": "street",
}

US_STATE_MAP = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

INDIA_STATE_MAP = {
    "up": "uttar pradesh", "dl": "delhi", "mh": "maharashtra", "ka": "karnataka",
    "tn": "tamil nadu", "gj": "gujarat", "wb": "west bengal", "mp": "madhya pradesh",
    "hr": "haryana", "rj": "rajasthan", "ap": "andhra pradesh", "tg": "telangana",
    "ts": "telangana", "kl": "kerala", "pb": "punjab", "br": "bihar",
    "od": "odisha", "jh": "jharkhand", "cg": "chhattisgarh", "uk": "uttarakhand",
    "ua": "uttarakhand", "hp": "himachal pradesh", "jk": "jammu and kashmir",
    "as": "assam", "ga": "goa", "ch": "chandigarh",
}

FRANCE_STREET_MAP = {
    "r": "rue", "av": "avenue", "bd": "boulevard", "all": "allee",
    "pl": "place", "imp": "impasse", "rte": "route", "ch": "chemin",
}

APOSTROPHE_RE = re.compile(r"['’`]")
NON_WORD_RE = re.compile(r"[^\w\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

# C-level translation table mapping ASCII punctuation to spaces (apostrophes deleted)
_ASCII_TRANS = {i: " " for i in range(128) if not chr(i).isalnum() and chr(i) != " "}
_ASCII_TRANS[ord("'")] = None
_ASCII_TRANS[ord("_")] = " "
_PUNCT_TABLE = str.maketrans(_ASCII_TRANS)


def strip_accents(text: str) -> str:
    """Normalizes accented/diacritic characters to ASCII equivalents."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_general_text(text: str, is_name: bool = False) -> str:
    """General lowercase, Devanagari transliteration, accent & punctuation normalization."""
    if not text:
        return ""
    if not text.isascii():
        if "\u200b" in text or "\u200c" in text or "\u200d" in text or "\ufeff" in text:
            text = _ZW_CHARS_RE.sub("", text)
        text = transliterate_if_needed(text)
        text = strip_accents(text)
        if "’" in text or "`" in text:
            text = text.replace("’", "").replace("`", "")
    text = text.lower()

    if is_name and ("." in text or "/" in text):
        text = DOMAIN_SUFFIX_RE.sub(" ", text)
        text = URL_PREFIX_RE.sub(" ", text)

    # Collapse dotted/slashed initials before removing punctuation
    if "." in text or "/" in text:
        text = INITIALS_3_RE.sub(r"\1\2\3", text)
        text = INITIALS_2_RE.sub(r"\1\2", text)

    if "&" in text:
        text = text.replace("&", " and ")
    if "+" in text:
        text = text.replace("+", " plus ")
    if "@" in text:
        text = text.replace("@", " at ")

    if text.isascii():
        text = " ".join(text.translate(_PUNCT_TABLE).split())
    else:
        if "'" in text:
            text = text.replace("'", "")
        text = " ".join(NON_WORD_RE.sub(" ", text).split())

    if is_name and text:
        if " " not in text and text.endswith("com") and len(text) >= 9:
            m = CONCAT_COM_RE.match(text)
            if m:
                text = m.group(1)
        if "lndia" in text:
            text = text.replace("lndia", "india")
        # Fix single OCR digit inside alpha business name words (e.g. '6enuine' -> 'genuine', '8urn' -> 'burn', 'st0rage' -> 'storage')
        if any(c.isdigit() for c in text):
            fixed_toks = []
            for tok in text.split():
                if not tok.isdigit() and not tok.isalpha() and len(tok) >= 4:
                    n_dig = sum(1 for c in tok if c.isdigit())
                    if n_dig == 1 and not tok.endswith(("st", "nd", "rd", "th")):
                        tok = tok.translate(_OCR_WORD_DIGIT_TRANS)
                fixed_toks.append(tok)
            text = " ".join(fixed_toks)
        if " " in text:
            text = _DUP_BIGRAM_RE.sub(r"\1", text)
            text = _DUP_WORD_RE.sub(r"\1", text)

    return text


@lru_cache(maxsize=64_000)
def clean_business_name(name: str) -> Tuple[str, str]:
    """
    Returns (cleaned_full_name, name_stripped_of_legal_suffixes).
    Cached for fast repeated lookup across blocking and scoring.
    """
    cleaned_full = normalize_general_text(name, is_name=True)
    stripped = " ".join(LEGAL_SUFFIXES_REGEX.sub(" ", cleaned_full).split())
    if " " in stripped:
        no_hon = HONORIFIC_PREFIX_RE.sub("", stripped).strip()
        if no_hon:
            stripped = no_hon
    if any(w in stripped for w in ("private", "limited", "pvtltd", "limatid", "public")):
        stripped = CONCAT_LEGAL_SUFFIX_RE.sub(r"\1", stripped)
    if " " in stripped and any(ch.isdigit() for ch in stripped):
        no_reg = " ".join(LONG_REG_NUM_RE.sub(" ", stripped).split())
        if no_reg:
            stripped = no_reg
    if " " in stripped:
        stripped = _DUP_WORD_RE.sub(r"\1", stripped)
    return cleaned_full, (stripped if stripped else cleaned_full)


_SKEL_CIEY_RE = re.compile(r"c([iey])")
_SKEL_GIEY_RE = re.compile(r"g([iey])")
_SKEL_FINAL_Z_RE = re.compile(r"z\b")
_SKEL_VOWELS_RE = re.compile(r"[aeiouy]+")
_SKEL_REPEAT_RE = re.compile(r"(.)\1+")


@lru_cache(maxsize=128_000)
def phonetic_skeleton_word(word: str) -> str:
    """
    Computes a cross-script & typo-resilient phonetic consonant skeleton for a single word.
    Maps English and transliterated Indic spellings to an identical canonical form
    (e.g., 'sunrise' == 'sanrais' -> 'snrs', 'technologies' == 'teknalajis' -> 'tknljs').
    """
    if not word:
        return ""
    s = word.lower()
    s = s.replace("tion", "sn").replace("sion", "sn").replace("ture", "cr")
    s = s.replace("dge", "j").replace("dg", "j").replace("tch", "C")
    s = s.replace("wh", "v").replace("ph", "f").replace("bh", "v").replace("w", "v")
    s = s.replace("chh", "C").replace("ch", "C").replace("sh", "s")
    s = s.replace("ck", "k").replace("qu", "k").replace("q", "k").replace("x", "ks")
    if "c" in s:
        s = _SKEL_CIEY_RE.sub(r"s\1", s)
        s = s.replace("c", "k")
    if "C" in s:
        s = s.replace("C", "c")
    if "g" in s:
        s = _SKEL_GIEY_RE.sub(r"j\1", s)
        s = s.replace("gh", "g")
    s = s.replace("dh", "d").replace("th", "t").replace("kh", "k")
    s = s.replace("nb", "mb").replace("np", "mp")
    if s.endswith("z"):
        s = s[:-1] + "s"
    s = _SKEL_VOWELS_RE.sub("", s)
    if len(s) >= 4 and s.endswith("j"):
        s = s[:-1] + "s"
    s = _SKEL_REPEAT_RE.sub(r"\1", s)
    return s


@lru_cache(maxsize=64_000)
def phonetic_skeleton(text: str) -> str:
    """Returns space-separated phonetic consonant skeletons for all tokens in text."""
    if not text:
        return ""
    return " ".join(phonetic_skeleton_word(w) for w in text.split() if w)


@lru_cache(maxsize=64_000)
def extract_name_variants(name: str) -> Tuple[str, ...]:
    """
    Returns a tuple of stripped name variants.
    If the name contains 'd/b/a' or 'a/k/a', includes both the full stripped name
    and each individual part stripped of legal suffixes.
    """
    _, full_stripped = clean_business_name(name)
    if not name:
        return (full_stripped,)
    low = name.lower()
    if ("/" not in low and "dba" not in low and "d.b.a" not in low and "aka" not in low) or not DBA_SPLIT_RE.search(low):
        return (full_stripped,)

    parts = DBA_SPLIT_RE.split(name)
    variants = [full_stripped]
    for p in parts:
        p = p.strip()
        if p:
            _, s_p = clean_business_name(p)
            if s_p and s_p not in variants:
                variants.append(s_p)
    return tuple(variants)


_ADDR_NULL_TOKENS = {"null", "na"}


@lru_cache(maxsize=64_000)
def clean_address(address: str, country: str = "") -> str:
    """
    Standardizes address tokens, expands road/street/state abbreviations,
    normalizes leading zeroes and ordinals, and splits concatenated number+word tokens
    (e.g. '188BIS' -> '188 bis', '146Kailash' -> '146 kailash', '2Sheela' -> '2 sheela').
    """
    if not address:
        return ""
    cleaned = normalize_general_text(address, is_name=False)
    state_map = (
        US_STATE_MAP if country == "US"
        else (INDIA_STATE_MAP if country == "India" else (FRANCE_STREET_MAP if country == "France" else None))
    )

    out_tokens = []
    for t in cleaned.split():
        if t in _ADDR_NULL_TOKENS:
            continue
        if t.isdigit():
            out_tokens.append(t.lstrip("0") or "0")
        elif t.isalpha():
            if t == "ist":
                out_tokens.append("1")
                continue
            t = ADDR_ABBR_MAP.get(t, t)
            if state_map is not None:
                t = state_map.get(t, t)
            out_tokens.append(t)
        else:
            if len(t) > 2 and t.endswith(("st", "nd", "rd", "th")) and t[:-2].isdigit():
                out_tokens.append(t[:-2].lstrip("0") or "0")
            elif _TWO_ALPHA_RE.search(t):
                for p in _ALNUM_SPLIT_RE.findall(t):
                    if p in _ADDR_NULL_TOKENS:
                        continue
                    if p.isdigit():
                        out_tokens.append(p.lstrip("0") or "0")
                    else:
                        p = ADDR_ABBR_MAP.get(p, p)
                        if state_map is not None:
                            p = state_map.get(p, p)
                        out_tokens.append(p)
            else:
                if t[0] == "0":
                    t = t.lstrip("0") or "0"
                out_tokens.append(t)

    return " ".join(out_tokens)
