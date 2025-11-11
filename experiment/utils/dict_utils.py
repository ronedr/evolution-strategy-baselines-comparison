from dataclasses import fields, replace


def update_params(dc, updates: dict):
    names = {f.name for f in fields(dc)}
    safe = {k: v for k, v in updates.items() if k in names}
    if not safe:
        return dc
    return replace(dc, **safe)
