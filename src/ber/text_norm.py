"""Text normalisation for business names and addresses.

Design rules:
  * COUNTRY-AGNOSTIC: nothing here depends on the country label (France appears only in the test set).
  * Uses only the standard library (unicodedata) for accent stripping.
  * Abbreviation dictionaries are small and generic; extend them from what you see in the training data.
"""
import re
import unicodedata

# ---------- names ----------
NAME_ABBR = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited", "pvt": "private",
    "co": "company", "cie": "company", "intl": "international", "svcs": "services",
    "svc": "service", "mfg": "manufacturing", "assoc": "associates", "bros": "brothers",
    "dept": "department", "grp": "group", "natl": "national", "mgmt": "management",
    "tech": "technologies", "technology": "technologies",
}
# legal-form words and glue words removed to obtain the "core" name (compared after abbreviation expansion)
NAME_STOP = {
    "incorporated", "corporation", "company", "limited", "private", "llc", "llp", "lp", "plc",
    "pty", "gmbh", "sarl", "sas", "sasu", "eurl", "sa", "sca", "snc", "opc", "ltda",
    "and", "et", "the", "of", "de", "du", "des", "la", "le", "les", "d", "l",
}


def strip_accents(s):
    """Remove combining accents from LATIN letters only (keeps non-Latin scripts intact)."""
    out = []
    for c in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(c) and out and ord(out[-1]) < 0x250:
            continue
        out.append(c)
    return "".join(out)


def norm_basic(s):
    """lowercase, strip accents, '&' -> 'and', punctuation -> space, collapse whitespace."""
    s = strip_accents(s).lower().replace("&", " and ")
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_name(s):
    """Normalised name with abbreviations expanded (legal suffixes kept)."""
    return " ".join(NAME_ABBR.get(t, t) for t in norm_basic(s).split())


def core_name(s):
    """Name without legal-form / glue words (falls back to full normalised name if nothing is left)."""
    full = norm_name(s)
    toks = [t for t in full.split() if t not in NAME_STOP]
    return " ".join(toks) if toks else full


def acronym(s):
    """First letters of the tokens, e.g. 'international business machines' -> 'ibm'."""
    return "".join(t[0] for t in s.split())


# ---------- addresses ----------
ADDR_ABBR = {
    "r": "rue",  # French: "R" is a very common abbreviation for "Rue" (street), e.g. "46 R HENRI CARRITTE"
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bvd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court", "pl": "place",
    "sq": "square", "hwy": "highway", "pkwy": "parkway", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "apt": "apartment", "sec": "sector", "mkt": "market", "cir": "circle",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "rte": "route", "ext": "extension", "opp": "opposite", "nr": "near",
}
# "near SBI ATM", "opp. City Mall", "behind X" ... cut from the keyword to the end of the comma-segment
_LANDMARK = re.compile(r"\b(near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|close to)\b[^,;]*")
_POSTAL = re.compile(r"(?<!\d)\d{5,6}(?!\d)")   # US ZIP / France code postal (5 digits), India PIN (6 digits)
_NUMS = re.compile(r"\d+")


def addr_full(s):
    """Normalised address INCLUDING landmark phrases, abbreviations expanded."""
    return " ".join(ADDR_ABBR.get(t, t) for t in norm_basic(s).split())


def addr_core(s):
    """Normalised address with landmark clauses removed (falls back to the full address if empty)."""
    raw = strip_accents(s).lower()
    cut = _LANDMARK.sub(" ", raw)
    core = " ".join(ADDR_ABBR.get(t, t) for t in norm_basic(cut).split())
    return core if core else addr_full(s)


def postal_codes(s):
    """5/6-digit postal codes found in the RAW address."""
    return frozenset(_POSTAL.findall(s))


def numbers(s):
    """All digit groups in the RAW address minus the postal codes (house/plot/street numbers)."""
    return frozenset(_NUMS.findall(s)) - postal_codes(s)
