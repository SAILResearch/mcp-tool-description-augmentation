from .evaluator import (
    Evaluator,
    EvaluationResult,
    EvaluatorConfig
)

# Import optional helper function modules defensively so missing
# third-party dependencies do not break test execution.
try:  # pragma: no cover - optional dependency
    from .functions import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - optional dependency
    from .github.functions import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - optional dependency
    from .google_maps.functions import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - optional dependency
    from .yfinance.functions import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - optional dependency
    from .blender.functions import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - optional dependency
    from .playwright.functions import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - optional dependency
    from .google_search.functions import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - optional dependency
    from .notion.functions import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - optional dependency
    from .weather.functions import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "EvaluatorConfig"
]
