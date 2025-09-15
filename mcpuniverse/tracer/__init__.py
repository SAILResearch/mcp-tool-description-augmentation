from typing import TYPE_CHECKING

__all__ = ["Tracer"]

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .tracer import Tracer as Tracer


def __getattr__(name: str):  # pragma: no cover - simple proxy
    if name == "Tracer":
        from .tracer import Tracer as _Tracer

        return _Tracer
    raise AttributeError(name)
