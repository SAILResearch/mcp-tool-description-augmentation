"""LLM module containing various language model implementations."""

# The individual model classes depend on a number of optional third-party
# libraries.  Import them defensively so the absence of a dependency does
# not prevent the package from being imported.  Missing models are exposed
# as ``None`` which allows consumers to handle the situation gracefully.

try:  # pragma: no cover - optional dependency
    from .openai import OpenAIModel
except Exception:  # pragma: no cover
    OpenAIModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .mistral import MistralModel
except Exception:  # pragma: no cover
    MistralModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .claude import ClaudeModel
except Exception:  # pragma: no cover
    ClaudeModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .ollama import OllamaModel
except Exception:  # pragma: no cover
    OllamaModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .deepseek import DeepSeekModel
except Exception:  # pragma: no cover
    DeepSeekModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .claude_gateway import ClaudeGatewayModel
except Exception:  # pragma: no cover
    ClaudeGatewayModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .grok import GrokModel
except Exception:  # pragma: no cover
    GrokModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .openai_agent import OpenAIAgentModel
except Exception:  # pragma: no cover
    OpenAIAgentModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .openrouter import OpenRouterModel
except Exception:  # pragma: no cover
    OpenRouterModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .gemini import GeminiModel
except Exception:  # pragma: no cover
    GeminiModel = None  # type: ignore


__all__ = [
    "OpenAIModel",
    "MistralModel",
    "ClaudeModel",
    "OllamaModel",
    "DeepSeekModel",
    "ClaudeGatewayModel",
    "GrokModel",
    "OpenAIAgentModel",
    "OpenRouterModel",
    "GeminiModel",
]
