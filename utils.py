import json
from types import MappingProxyType

def deep_to_serializable(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return "<circular reference>"
    seen.add(obj_id)

    if isinstance(obj, MappingProxyType):
        obj = dict(obj)

    if isinstance(obj, dict):
        return {str(k): deep_to_serializable(v, seen) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [deep_to_serializable(item, seen) for item in obj]
    elif hasattr(obj, "__dict__"):
        return deep_to_serializable(vars(obj), seen)
    else:
        try:
            json.dumps(obj)  # test if serializable
            return obj
        except TypeError:
            return str(obj)  # fallback

def pretty_print(obj):
    print(json.dumps(deep_to_serializable(obj), indent=2))
