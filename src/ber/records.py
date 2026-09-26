"""Stack the three sources into ONE table with normalised columns. Row position == matrix row everywhere."""
import pandas as pd
from . import text_norm as tn


def build_records(s1, s2, s3):
    """Return (rec, id2row). rec has RangeIndex 0..N-1 and columns:
    entity_id, source (S1/S2/S3), country, country_key, name/address raw + normalised variants."""
    rec = pd.concat(
        [s1.assign(source="S1"), s2.assign(source="S2"), s3.assign(source="S3")], ignore_index=True
    )
    rec["country_key"] = rec["country"].str.strip().str.lower()
    rec["name_norm"] = rec["business_name"].map(tn.norm_name)
    rec["name_core"] = rec["business_name"].map(tn.core_name)
    rec["name_acr"] = rec["name_core"].map(tn.acronym)
    rec["addr_core"] = rec["business_address"].map(tn.addr_core)
    rec["postal"] = rec["business_address"].map(tn.postal_codes)
    rec["nums"] = rec["business_address"].map(tn.numbers)
    rec["name_addr"] = rec["name_core"] + " | " + rec["addr_core"]
    id2row = dict(zip(rec["entity_id"], rec.index))
    return rec, id2row
