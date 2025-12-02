from dataclasses import fields, replace
import json
import hashlib

def update_params(dc, updates: dict):
    names = {f.name for f in fields(dc)}
    safe = {k: v for k, v in updates.items() if k in names}
    if not safe:
        return dc
    return replace(dc, **safe)


def to_unique_hash(d: dict) -> str:
    """
    Return a SHA256 hash of a dictionary.
    Same dictionary content -> same hash, regardless of key order.
    """
    # Make a canonical JSON representation: sorted keys, no extra spaces
    encoded = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
