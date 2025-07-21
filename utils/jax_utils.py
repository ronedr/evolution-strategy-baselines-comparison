import jax
import numpy as np


def to_numpy(obj):
    if isinstance(obj, jax.Array):
        return np.array(obj).tolist()  # Moves data to host and detaches from device
    elif isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(v) for v in obj)
    else:
        return obj.tolist()
