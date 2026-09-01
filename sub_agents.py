"""LLM conversational specialists, configured for a local Helix endpoint."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.registry import LLMRegistry


load_dotenv()


class HelixGemini(LiteLlm):
    """Routes ``helix/<model>`` names to Helix's OpenAI-compatible local API."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        if not model.startswith("helix/"):
            raise ValueError("HelixGemini model names must begin with 'helix/'.")
        helix_model = model.removeprefix("helix/")
        super().__init__(
            # LiteLLM's OpenAI adapter sends the unmodified model ID to Helix.
            model=f"openai/{helix_model}",
            api_base=os.getenv("HELIX_API_BASE", "http://localhost:8080/v1"),
            api_key=os.getenv("HELIX_API_KEY", "local-helix"),
            **kwargs,
        )

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"helix/.*"]


# Required before ADK resolves the model string used by the LLM agents below.
LLMRegistry.register(HelixGemini)

HELIX_MODEL = os.getenv("HELIX_MODEL", "helix/gemini-2.5-flash")

greeting_agent = LlmAgent(
    name="greeting_agent",
    model=HELIX_MODEL,
    instruction="Greet an inbound wire-fraud caller warmly. Ask only for the next authentication detail.",
)

intent_classifier_agent = LlmAgent(
    name="intent_classifier_agent",
    model=HELIX_MODEL,
    instruction=(
        "Classify the caller's reason as exactly one label: WIRE_FRAUD, OUT_OF_SCOPE, "
        "or NEEDS_HUMAN. Return only that label. Never make account or transaction changes."
    ),
    output_schema=str,
)

transaction_review_agent = LlmAgent(
    name="transaction_review_agent",
    model=HELIX_MODEL,
    instruction=(
        "Present supplied wire transactions one at a time. Ask whether the caller recognizes each. "
        "Do not claim a transaction is fraudulent; collect the caller's answer."
    ),
)

ato_interview_agent = LlmAgent(
    name="ato_interview_agent",
    model=HELIX_MODEL,
    instruction=(
        "Ask concise account-takeover questions about unknown logins, password resets, "
        "new devices, or unexpected contact-detail changes. Collect answers without changing data."
    ),
)

contact_verification_agent = LlmAgent(
    name="contact_verification_agent",
    model=HELIX_MODEL,
    instruction="Verify the caller's phone, email, and address. Collect corrections clearly; do not update records yourself.",
)

closing_agent = LlmAgent(
    name="closing_agent",
    model=HELIX_MODEL,
    instruction="Give a short, empathetic closing after a wire-fraud case is routed. Do not promise recovery or a specific outcome.",
)
