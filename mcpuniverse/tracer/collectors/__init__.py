from typing import TYPE_CHECKING, Any

__all__ = [
    "MemoryCollector",
    "SQLiteCollector",
    "FileCollector",
    "BaseCollector",
]

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .memory import MemoryCollector as MemoryCollector
    from .sqlite import SQLiteCollector as SQLiteCollector
    from .file import FileCollector as FileCollector
    from .base import BaseCollector as BaseCollector


def __getattr__(name: str) -> Any:  # pragma: no cover - simple proxy
    if name == "MemoryCollector":
        from .memory import MemoryCollector as _MemoryCollector

        return _MemoryCollector
    if name == "SQLiteCollector":
        from .sqlite import SQLiteCollector as _SQLiteCollector

        return _SQLiteCollector
    if name == "FileCollector":
        from .file import FileCollector as _FileCollector

        return _FileCollector
    if name == "BaseCollector":
        from .base import BaseCollector as _BaseCollector

        return _BaseCollector
    raise AttributeError(name)
