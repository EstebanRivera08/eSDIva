import numpy as np


def mat_struct_fields(ms):
    if hasattr(ms, "_fieldnames"):
        return list(ms._fieldnames)
    if hasattr(ms, "__dict__"):
        return [k for k in vars(ms).keys() if not k.startswith("_")]
    return []


def mat_struct_to_dict(ms):
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
