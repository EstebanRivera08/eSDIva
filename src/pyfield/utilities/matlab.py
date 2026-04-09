"""Utilities for reading and exploring MATLAB .mat file structures."""

import numpy as np


def mat_struct_fields(ms):
    """Return field names from a MATLAB struct-like object.

    Parameters
    ----------
    ms : object
        MATLAB struct loaded via ``scipy.io.loadmat``.

    Returns
    -------
    list of str
        Field names of the struct.
    """
    if hasattr(ms, "_fieldnames"):
        return list(ms._fieldnames)
    if hasattr(ms, "__dict__"):
        return [k for k in vars(ms).keys() if not k.startswith("_")]
    return []


def mat_struct_to_dict(ms):
    """Convert a MATLAB struct recursively to a Python dict.

    Parameters
    ----------
    ms : object
        MATLAB struct or numpy object array to convert.

    Returns
    -------
    dict or list or object
        Converted Python representation.
    """
    # recursive conversion: mat_struct -> dict, object arrays -> lists
    if isinstance(ms, np.ndarray) and ms.dtype == object:
        return [mat_struct_to_dict(x) for x in ms.flat]
    if hasattr(ms, "_fieldnames"):
        d = {}
        for fn in ms._fieldnames:
            d[fn] = mat_struct_to_dict(getattr(ms, fn))
        return d
    if hasattr(ms, "__dict__"):
        return {
            k: mat_struct_to_dict(v)
            for k, v in vars(ms).items()
            if not k.startswith("_")
        }
    return ms


def explore_mat(obj, name="root", depth=0, max_depth=3):
    """Print the hierarchical structure of a MATLAB .mat object.

    Parameters
    ----------
    obj : object
        MATLAB object to explore.
    name : str, optional
        Display name for the root node.
    depth : int, optional
        Current recursion depth.
    max_depth : int, optional
        Maximum recursion depth.
    """
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{name}: {type(obj)} (max depth)")
        return
    if hasattr(obj, "_fieldnames") or hasattr(obj, "__dict__"):
        print(f"{indent}{name}: MATLAB struct fields -> {mat_struct_fields(obj)}")
        for fn in mat_struct_fields(obj):
            explore_mat(
                getattr(obj, fn),
                name=f"{name}.{fn}",
                depth=depth + 1,
                max_depth=max_depth,
            )
    elif isinstance(obj, np.ndarray):
        print(f"{indent}{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
        if obj.dtype == object and obj.size:
            for i, el in enumerate(obj.flat[:5]):
                explore_mat(
                    el, name=f"{name}[{i}]", depth=depth + 1, max_depth=max_depth
                )
    else:
        print(f"{indent}{name}: {type(obj)} value={repr(obj)[:200]}")
