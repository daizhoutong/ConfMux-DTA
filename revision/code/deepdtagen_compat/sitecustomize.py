import numpy as np

# ------------------------------------------------------------
# NumPy compatibility for older fairseq
# ------------------------------------------------------------
_aliases = {
    "float": float,
    "int": int,
    "complex": complex,
    "bool": bool,
    "object": object,
    "str": str,
    "long": int,
}

for _name, _value in _aliases.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


# ------------------------------------------------------------
# Python 3.11 dataclasses compatibility for older fairseq
#
# Python 3.11 rejects unhashable dataclass-instance defaults.
# Older fairseq used these defaults and imported successfully
# under older Python versions.
#
# Only relax this rule for fairseq-owned config objects.
# ------------------------------------------------------------
import dataclasses

_original_get_field = dataclasses._get_field


def _compat_get_field(cls, a_name, a_type, default_kw_only):
    try:
        return _original_get_field(
            cls, a_name, a_type, default_kw_only
        )
    except ValueError as exc:
        msg = str(exc)

        if (
            cls.__module__.startswith("fairseq")
            and "mutable default" in msg
            and "default_factory" in msg
        ):
            default = getattr(cls, a_name, dataclasses.MISSING)

            if default is not dataclasses.MISSING:
                default_cls = default.__class__

                # Only patch fairseq's own config/dataclass objects.
                if default_cls.__module__.startswith("fairseq"):
                    old_hash = getattr(default_cls, "__hash__", None)

                    try:
                        default_cls.__hash__ = object.__hash__
                        return _original_get_field(
                            cls, a_name, a_type, default_kw_only
                        )
                    finally:
                        default_cls.__hash__ = old_hash

        raise


dataclasses._get_field = _compat_get_field
