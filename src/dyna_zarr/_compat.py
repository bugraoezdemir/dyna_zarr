"""Renamed parameters: the old names keep working for a while, with a warning."""
import warnings


def renamed(value, old_value, *, new, old, where, default=None, stacklevel=3):
    """The value of a renamed parameter.

    ``value`` was passed as ``new`` (e.g. ``max_threads``), ``old_value`` as its old
    name ``old`` (e.g. ``max_workers``). The old name works, with a FutureWarning
    naming the new one; passing both is an error. Neither: ``default``.
    """
    if old_value is None:
        return default if value is None else value
    if value is not None:
        raise TypeError(f"{where}: {old}= is the old name of {new}=; pass {new}= only")
    warnings.warn(f"{where}: {old}= is deprecated; it is now called {new}= (same "
                  f"meaning)", FutureWarning, stacklevel=stacklevel)
    return old_value
