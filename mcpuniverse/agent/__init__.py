"""Agent implementations with optional dependencies."""

# Import each agent defensively so missing optional dependencies do not
# prevent the package from being imported during tests.

try:  # pragma: no cover - optional dependency
    from .function_call import FunctionCall
except Exception:  # pragma: no cover
    FunctionCall = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .basic import BasicAgent
except Exception:  # pragma: no cover
    BasicAgent = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .workflow import WorkflowAgent
except Exception:  # pragma: no cover
    WorkflowAgent = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .react import ReAct
except Exception:  # pragma: no cover
    ReAct = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .reflection import Reflection
except Exception:  # pragma: no cover
    Reflection = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .explore_and_exploit import ExploreAndExploit
except Exception:  # pragma: no cover
    ExploreAndExploit = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .base import BaseAgent
except Exception:  # pragma: no cover
    BaseAgent = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .claude_code import ClaudeCodeAgent
except Exception:  # pragma: no cover
    ClaudeCodeAgent = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .openai_agent_sdk import OpenAIAgentSDK
except Exception:  # pragma: no cover
    OpenAIAgentSDK = None  # type: ignore


__all__ = [
    "FunctionCall",
    "BasicAgent",
    "WorkflowAgent",
    "ReAct",
    "Reflection",
    "ExploreAndExploit",
    "BaseAgent",
    "ClaudeCodeAgent",
    "OpenAIAgentSDK",
]
