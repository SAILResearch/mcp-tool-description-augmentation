"""
Chat-completions compatible vLLM client for SAIL Lab deployments.
"""
# pylint: disable=broad-exception-caught
import json
import os
import time
import logging
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Union, Optional, Type, List, Any

import requests
from openai import RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel

from mcpuniverse.common.config import BaseConfig
from mcpuniverse.common.context import Context
from .base import BaseLLM

load_dotenv()


@dataclass
class VLLMSailLabConfig(BaseConfig):
    """
    Configuration for SAIL Lab-hosted vLLM chat models.

    Attributes:
        model_name (str): The model identifier served by the vLLM instance.
        provider (str): Provider identifier expected by the server (if any).
        base_url (str): Base URL of the vLLM API.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        frequency_penalty (float): Penalizes frequent tokens.
        presence_penalty (float): Penalizes repeated topics.
        max_completion_tokens (int): Maximum tokens to generate.
        seed (int): Random seed for reproducibility.
    """

    model_name: str = "qwen80b"
    provider: str = ""
    base_url: str = os.environ.get("VLLM_SAIL_LAB_BASE_URL", "http://otto.cs.queensu.ca:8002")
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_completion_tokens: int = 2048
    seed: int = 12345


class VLLMSailLabModel(BaseLLM):
    """
    Chat-completions client for SAIL Lab vLLM deployments.
    """

    config_class = VLLMSailLabConfig
    alias = "vllm_sail_lab"
    env_vars = ["VLLM_SAIL_LAB_BASE_URL"]

    def __init__(self, config: Optional[Union[Dict, str]] = None):
        super().__init__()
        self.config = VLLMSailLabModel.config_class.load(config)

    def _generate(
        self,
        messages: List[dict[str, str]],
        response_format: Type[PydanticBaseModel] = None,
        **kwargs,
    ):
        """
        Generates chat completions using the SAIL Lab vLLM endpoint.
        """

        max_retries = kwargs.get("max_retries", 5)
        base_delay = kwargs.get("base_delay", 10.0)
        _ = response_format

        tools: Optional[List[Dict[str, Any]]] = kwargs.get("tools")
        tool_choice: Optional[Dict[str, Any]] = kwargs.get("tool_choice")

        effective_messages = deepcopy(messages)

        for attempt in range(max_retries + 1):
            try:
                payload: Dict[str, Any] = {
                    "model": self.config.model_name,
                    "messages": effective_messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_completion_tokens,
                    "top_p": self.config.top_p,
                    "frequency_penalty": self.config.frequency_penalty,
                    "presence_penalty": self.config.presence_penalty,
                    "seed": self.config.seed,
                }
                if self.config.provider:
                    payload["provider"] = self.config.provider
                if tools:
                    payload["tools"] = tools
                if tool_choice:
                    payload["tool_choice"] = tool_choice

                response = requests.post(
                    f"{self.config.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=int(kwargs.get("timeout", 600)),
                )
                try:
                    response.raise_for_status()
                except Exception as http_err:  # requests raises HTTPError here
                    logging.error(
                        "HTTP error from SAIL Lab vLLM: %s | body=%s",
                        http_err,
                        response.text,
                    )

                    if (
                        tools
                        and response.status_code == 400
                        and not kwargs.get("_tools_text_fallback")
                    ):
                        logging.warning(
                            "Retrying without OpenAI tool payload; embedding tool descriptions into the prompt instead."
                        )
                        effective_messages = self._inject_tool_descriptions(
                            effective_messages, tools
                        )
                        tools = None
                        kwargs["_tools_text_fallback"] = True
                        continue

                    raise

                response_json = response.json()
                choices = response_json.get("choices", [])
                if not choices:
                    return None

                message = choices[0].get("message", {})
                content = message.get("content")
                tool_calls = message.get("tool_calls")

                usage = response_json.get("usage", {}) or {}
                usage_obj = SimpleNamespace(**usage) if usage else None

                message_obj = SimpleNamespace(content=content, tool_calls=tool_calls)
                choice_obj = SimpleNamespace(message=message_obj)
                response_obj = SimpleNamespace(
                    choices=[choice_obj],
                    model=response_json.get("model", self.config.model_name),
                    usage=usage_obj,
                )

                if tools or tool_calls:
                    return response_obj

                usage_summary = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                return response_obj if tool_calls else content or ""

            except (RateLimitError, APIError, APITimeoutError) as e:
                if attempt == max_retries:
                    logging.warning(
                        "All %d attempts failed. Last error: %s", max_retries + 1, e
                    )
                    return None

                delay = base_delay * (2 ** attempt)
                logging.info(
                    "Attempt %d failed with error: %s. Retrying in %.1f seconds...",
                    attempt + 1,
                    e,
                    delay,
                )
                time.sleep(delay)

            except Exception as e:
                logging.error("Non-retryable error occurred: %s", e)
                return None

    def _inject_tool_descriptions(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Embed tool descriptions directly into the prompt for servers that reject native tool payloads."""

        tool_lines: List[str] = [
            "You have access to the following tools. Use them by responding with a JSON function call if needed."
        ]

        for tool in tools:
            function = tool.get("function", {})
            name = function.get("name", "unknown_tool")
            description = function.get("description", "")
            parameters = function.get("parameters", {})
            pretty_schema = json.dumps(parameters, indent=2, ensure_ascii=False)

            tool_lines.append(
                f"- Tool `{name}`: {description}\n  Parameters schema:\n{pretty_schema}"
            )

        return ([{"role": "system", "content": "\n".join(tool_lines)}] + messages)

    def set_context(self, context: Context):
        """Set runtime context from the benchmark runner."""

        super().set_context(context)
        self.config.base_url = context.env.get(
            "VLLM_SAIL_LAB_BASE_URL", self.config.base_url
        )

    def support_tool_call(self) -> bool:
        """Return a flag indicating tool/function call support."""

        return True
