"""Calendar and deterministic company splits, independent of labels."""
import hashlib
import numpy as np
from .common import object_hash

DEVELOPMENT = [("D1",2005,2006,2007),("D2",2007,2008,2009),("D3",2009,2010,2011),("D4",2011,2012,2012)]

def group_id(company):
    return int(hashlib.sha256(("distress-group-stress-v1"+str(company)).encode()).hexdigest(),16)%3

def groups(df):
    return df.company_name.map(group_id).to_numpy()

def tie_key(company, year):
    return hashlib.sha256(f"{company}|{year}".encode()).hexdigest()

def row_hash(df):
    return object_hash(sorted(map(int,df.row_id)))

def inner_masks(df):
    cutoff = int(df.year.max())
    return (df.year < cutoff).to_numpy(), (df.year == cutoff).to_numpy()

def assert_disjoint(*frames):
    seen=set()
    for frame in frames:
        ids=set(frame.row_id)
        assert not (seen & ids)
        seen |= ids
